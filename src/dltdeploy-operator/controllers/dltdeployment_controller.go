/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"strconv"
	"strings"
	"time"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1alpha1 "github.com/VUZhuangweiKang/CNDLSys/tree/main/src/dltdeploy-operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	APIVersion    = "core/v1"
	PodKind       = "Pod"
	ConfigMapKind = "ConfigMap"
	ClientImage   = "zhuangweikang/cndlsys-dev:client"
)

// DLTDeploymentReconciler reconciles a DLTDeployment object
type DLTDeploymentReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=docgroup.com,resources=dltdeployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=docgroup.com,resources=dltdeployments/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=docgroup.com,resources=dltdeployments/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *DLTDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("start reconciling")
	rand.Seed(time.Now().UnixNano())

	var dltdeploy v1alpha1.DLTDeployment

	createWorkers := func(j int) error {
		job := &dltdeploy.Spec.Jobs[j]
		numWorkers := job.NumWorkers

		// create training workers (Pods)
		var masterIP string
		workers := make([]string, numWorkers)
		for w := 0; w < numWorkers; w++ {
			if pod, err := r.CreateWorkerPod(ctx, string(dltdeploy.Name), job, w); err != nil {
				return err
			} else {
				if w > 0 {
					for i := 0; i < len(pod.Spec.Containers); i++ {
						if masterIP == "" {
							panic("MASTER_ADDR is empty")
						}
						pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
							Name:  "MASTER_ADDR",
							Value: masterIP,
						})
					}
				}
				if err = ctrl.SetControllerReference(&dltdeploy, pod, r.Scheme); err != nil {
					logger.Error(err, fmt.Sprintf("error in assigning pod %s: %s.", pod.Name, err.Error()))
					return err
				} else {
					if err = r.Create(ctx, pod, &client.CreateOptions{}); err != nil {
						logger.Error(err, fmt.Sprintf("error in creating pod %s: %s.", pod.Name, err.Error()))
						return err
					}
				}
				logger.Info(fmt.Sprintf("creating worker: %s", pod.Name))
				workers[w] = pod.Name
				// waiting for pod be ready
				if w == 0 {
					for {
						pod = &corev1.Pod{}
						if err := r.Get(ctx, types.NamespacedName{Namespace: req.Namespace, Name: workers[w]}, pod); err != nil {
							time.Sleep(1 * time.Second)
						}
						if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
							masterIP = pod.Status.PodIP
							break
						}
						time.Sleep(1 * time.Second)
					}
				}
			}
		}
		job.Workers = workers
		return nil
	}

	updateWorkers := func(j int) error {
		var gracePeriodSec int64 = 0
		// delete the old pods
		for _, worker := range dltdeploy.Spec.Jobs[j].Workers {
			logger.Info(fmt.Sprintf("updating worker %s", worker))
			for {
				oldPod := corev1.Pod{}
				if err := r.Get(ctx, types.NamespacedName{Namespace: req.Namespace, Name: worker}, &oldPod); err != nil {
					logger.Error(err, fmt.Sprintf("error in getting pod %s", req.Name))
					break
				} else if err := r.Delete(ctx, &oldPod, &client.DeleteOptions{GracePeriodSeconds: &gracePeriodSec}); err != nil {
					continue
				}
			}
		}

		// create new pods
		if err := createWorkers(j); err != nil {
			return err
		}
		return nil
	}

	// one configmap for each job
	createConfigMap := func(j int) corev1.ConfigMap {
		job := dltdeploy.Spec.Jobs[j]
		name := fmt.Sprintf("%s-%s", dltdeploy.Name, job.UID)
		configmap := corev1.ConfigMap{
			TypeMeta:   metav1.TypeMeta{Kind: ConfigMapKind, APIVersion: APIVersion},
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: dltdeploy.Namespace},
			Data:       map[string]string{},
		}
		jobInfo := map[string]interface{}{
			"name":       job.Name,
			"uid":        job.UID,
			"workers":    job.Workers,
			"nodes":      job.Nodes,
			"datasource": job.DataSource,
			"credential": dltdeploy.Spec.Credential,
		}
		byteArr, _ := json.Marshal(jobInfo)
		configmap.Data["jobInfo.json"] = string(byteArr)

		err := ctrl.SetControllerReference(&dltdeploy, &configmap, r.Scheme)
		checkErr(err)
		return configmap
	}

	if err := r.Get(ctx, req.NamespacedName, &dltdeploy); err != nil {
		if k8serrors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			return ctrl.Result{}, nil
		} else {
			// Error reading the object - requeue the request.
			logger.Error(err, fmt.Sprintf("Trying to get the dltdeploy %s", dltdeploy.Name))
			return ctrl.Result{}, err
		}
	} else {
		if dltdeploy.DeletionTimestamp != nil {
			logger.Error(err, "DeletionTimestamp is not Nil")
			return ctrl.Result{}, err
		}
		spec := dltdeploy.Spec
		for j := 0; j < len(spec.Jobs); j++ {
			oldConfigMap := &corev1.ConfigMap{}
			if err := r.Get(ctx, types.NamespacedName{Namespace: req.Namespace, Name: fmt.Sprintf("%s-%s", dltdeploy.Name, spec.Jobs[j].UID)}, oldConfigMap); err != nil {
				if k8serrors.IsNotFound(err) {
					newConfigMap := createConfigMap(j)
					if err := r.Create(ctx, &newConfigMap, &client.CreateOptions{}); err != nil {
						logger.Error(err, fmt.Sprintf("error in creating configmap %s: %s.", newConfigMap.Name, err.Error()))
						return ctrl.Result{}, err
					}
					if err := createWorkers(j); err != nil {
						logger.Error(err, "Failed to create workers for job")
						return ctrl.Result{}, err
					}
				} else {
					logger.Error(err, "Unkown error when getting the configmap")
					return ctrl.Result{}, err
				}
			} else {
				// update configmap
				oldSpec := v1alpha1.DLTDeploymentSpec{}
				if err := json.Unmarshal([]byte(dltdeploy.Annotations["spec"]), &oldSpec); err != nil {
					return ctrl.Result{}, err
				}
				if !reflect.DeepEqual(spec, oldSpec) {
					newConfigMap := createConfigMap(j)
					if err := r.Update(ctx, &newConfigMap, &client.UpdateOptions{}); err != nil {
						logger.Error(err, fmt.Sprintf("error in updating configmap %s", req.Name))
						return ctrl.Result{}, err
					}
					if err := updateWorkers(j); err != nil {
						logger.Error(err, "failed to update workers for job")
						return ctrl.Result{}, err
					}
				}
				return ctrl.Result{}, nil
			}
		}
	}

	// attach Annotations
	data, _ := json.Marshal(dltdeploy.Spec)
	if dltdeploy.Annotations != nil {
		dltdeploy.Annotations["spec"] = string(data)
	} else {
		dltdeploy.Annotations = map[string]string{"spec": string(data)}
	}
	if err := r.Update(ctx, &dltdeploy, &client.UpdateOptions{}); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

func (r *DLTDeploymentReconciler) FindUsableNodes(ctx context.Context) []corev1.Node {
	allNodes := &corev1.NodeList{}
	err := r.List(ctx, allNodes, &client.ListOptions{})
	checkErr(err)
	var nodes []corev1.Node
	activeNodeStatus := map[corev1.NodeConditionType]corev1.ConditionStatus{
		"NetworkUnavailable": "False",
		"PIDPressure":        "False",
		"DiskPressure":       "False",
		"MemoryPressure":     "False",
		"Ready":              "True",
	}
	for _, node := range allNodes.Items {
		passStatusCheck := true
		for _, condition := range node.Status.Conditions {
			if condition.Status != activeNodeStatus[condition.Type] {
				passStatusCheck = false
			}
		}
		if passStatusCheck {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

func (r *DLTDeploymentReconciler) CreateWorkerPod(ctx context.Context, dltdeployID string, job *v1alpha1.Job, workerIndex int) (*corev1.Pod, error) {
	workerID := fmt.Sprintf("%s-%s-%d", dltdeployID, job.UID, workerIndex)
	hostPathDir := corev1.HostPathDirectory
	volumes := []corev1.Volume{
		{
			Name:         "meta",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: fmt.Sprintf("%s-%s", dltdeployID, job.UID)}}},
		},
		{
			Name:         "share",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		},
		{
			Name: "shmem",
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
				Path: "/dev/shm",
				Type: &hostPathDir,
			}},
		},
		{
			Name:         "runtime",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory}},
		},
		{
			Name: "cndlsys-config",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "cndlsys-config"},
			}},
		},
	}
	mountPropagationMode := corev1.MountPropagationMode("HostToContainer")
	volMounts := []corev1.VolumeMount{
		{Name: "meta", MountPath: "/meta"},
		{Name: "share", MountPath: "/share"},
		{Name: "runtime", MountPath: "/runtime"},
		{Name: "shmem", MountPath: "/dev/shm"},
		{Name: "cndlsys-config", MountPath: "/configs", MountPropagation: &mountPropagationMode},
	}

	// for exp only, to test remote I/O performance
	selectedNodeAddr := job.Nodes[workerIndex]
	// selectedNodeAddr := "10.140.81.235"
	var selectedNodeHostName string
	nodes := r.FindUsableNodes(ctx)
	mountVols := func(device string) {
		for _, node := range nodes {
			nodeAddr := node.Status.Addresses[0].Address
			volName := strings.ReplaceAll(nodeAddr, ".", "-")
			volName = fmt.Sprintf("%s-%s", volName, device)
			// for local NFS server, we directly mount to the NFS mount point
			hostPathDirectoryOrCreate := corev1.HostPathDirectoryOrCreate
			if nodeAddr == selectedNodeAddr {
				selectedNodeHostName = node.Status.Addresses[1].Address
				volumes = append(volumes, corev1.Volume{
					Name: volName,
					VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
						Path: fmt.Sprintf("/nfs/%s", device),
						Type: &hostPathDirectoryOrCreate,
					}},
				})
			} else {
				volumes = append(volumes, corev1.Volume{
					Name: volName,
					VolumeSource: corev1.VolumeSource{NFS: &corev1.NFSVolumeSource{
						Server:   nodeAddr,
						Path:     fmt.Sprintf("/nfs/%s", device),
						ReadOnly: false,
					}},
				})
			}

			volMounts = append(volMounts, corev1.VolumeMount{
				Name:      volName,
				MountPath: fmt.Sprintf("/mnt/nfs/%s/%s", device, nodeAddr),
			})
		}
	}
	mountVols("ssd")
	mountVols("hdd")
	if len(selectedNodeHostName) == 0 {
		return nil, &customError{message: fmt.Sprintf("Specified node %s for the %d-th worker of job %s is unavailable.", selectedNodeAddr, workerIndex, job.UID)}
	}
	var containers []corev1.Container
	commonEnv := []corev1.EnvVar{
		{
			Name:      "NODE_IP",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.hostIP"}},
		},
		{
			Name:      "LOCAL_ADDR",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}},
		},
		{
			Name:  "MASTER_PORT",
			Value: "23456",
		},
		{
			Name:  "RANK",
			Value: strconv.Itoa(workerIndex),
		},
		{
			Name:  "WORLD_SIZE",
			Value: strconv.Itoa(job.NumWorkers),
		},
		{
			Name:  "MERGE",
			Value: strconv.Itoa(job.Merge),
		},
		{
			Name:  "PROBE",
			Value: strconv.Itoa(job.Probe),
		},
		{
			Name:  "THREADPOOLSIZE",
			Value: strconv.Itoa(job.ThreadPoolSize),
		},
	}
	if workerIndex == 0 {
		commonEnv = append(commonEnv, corev1.EnvVar{
			Name:      "MASTER_ADDR",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}},
		})
	}

	workerContainer := job.WorkerContainer
	workerContainer.Name = workerID
	workerContainer.Env = append(workerContainer.Env, commonEnv...)
	workerContainer.Resources = corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			"cpu":               resource.MustParse("8"),
			"memory":            resource.MustParse("18G"),
			"ephemeral-storage": resource.MustParse("20G"),
			// "nvidia.com/gpu":    resource.MustParse("4"),
		},
		Requests: corev1.ResourceList{
			"cpu":               resource.MustParse("8"),
			"memory":            resource.MustParse("18G"),
			"ephemeral-storage": resource.MustParse("20G"),
			// "nvidia.com/gpu":    resource.MustParse("4"),
		},
	}
	workerContainer.VolumeMounts = volMounts
	workerContainer.ImagePullPolicy = corev1.PullAlways

	containers = append(containers, workerContainer)
	clientContainer := corev1.Container{
		Name:            fmt.Sprintf("%s-client", workerID),
		Image:           ClientImage,
		ImagePullPolicy: corev1.PullAlways,
		WorkingDir:      "/app",
		Command:         []string{"python3", "Client.py"},
		// Command:      []string{"/bin/bash"},
		Env:          commonEnv,
		VolumeMounts: volMounts,
		TTY:          true,
		Stdin:        true,
		// Resources: corev1.ResourceRequirements{
		// 	Limits: corev1.ResourceList{
		// 		"cpu":               resource.MustParse("1"),
		// 		"memory":            resource.MustParse("2G"),
		// 		"ephemeral-storage": resource.MustParse("2G"),
		// 	},
		// },
	}
	containers = append(containers, clientContainer)
	if job.Probe == 1 {
		workerID = fmt.Sprintf("%s.%d", workerID, rand.Intn(1000))
	}
	pod := corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       PodKind,
			APIVersion: APIVersion,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      workerID,
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Volumes:       volumes,
			Containers:    containers,
			RestartPolicy: corev1.RestartPolicyNever,
			NodeName:      selectedNodeHostName,
		},
	}
	return &pod, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DLTDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DLTDeployment{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}

type customError struct {
	message string
}

func (e *customError) Error() string {
	return e.message
}

func checkErr(err error) {
	if err != nil {
		panic(err.Error())
	}
}
