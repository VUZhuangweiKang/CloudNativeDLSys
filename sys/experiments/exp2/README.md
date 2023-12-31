# Overview
[![DOI](https://zenodo.org/badge/387395302.svg)](https://zenodo.org/badge/latestdoi/387395302)

This repository contains public releases of [SenseTime](https://www.sensetime.com) Helios traces for the benefit of the deep learning system research community. 
<!-- Note that [Git LFS](https://git-lfs.github.com/) is required for downloading Helios traces. -->

If you do use the Helios traces in your research, please make sure to cite our SC '21 paper ["Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters"](https://doi.org/10.1145/3458817.3476223), which includes a comprehensive analysis of the deep learning workloads in Helios from April 2020 to September 2020.

We encourage anyone to use the traces for academic purposes, and if you had any questions, feel free to send an email to us, or file an issue on Github. 

<!-- > **Note that only the `Venus` trace is public available now. Other traces are being censored. We will release them as soon as possible.** -->

## Helios Description

Helios is a private datacenter dedicated to developing DL models for research and production in SenseTime. It contains 8 independent GPU clusters and over 12,000 GPUs in total.

In this repository, we publicly release the workload trace in 4 representative GPU clusters: *Earth*, *Saturn*, *Uranus*, and *Venus*. You can find a detailed description of the Helios datacenter in the SC '21 paper mentioned above. 

Besides, we also release the analysis scripts for Helios traces in [HeliosArtifact](https://github.com/S-Lab-System-Group/HeliosArtifact).

# Helios Dataset

The main trace characteristics, dataset structure and schema are:

## Main Characteristics:
*	Dataset size: 343MB
*   Compressed dataset size: 36.4MB
*	Number of files: 8
*	Duration: 6 months
*   Number of independent GPU clusters: 4
*	Total number of jobs: 3,362,981
*	Total number of GPU jobs: 1,580,464

## Dataset Structure

Each cluster provides a job trace file (`cluster_log.csv`) and a VC configuration file (`cluster_gpu_number.csv`).

```
📦data
 ┣ 📂Earth
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┣ 📂Saturn
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┣ 📂Uranus
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┗ 📂Venus
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
```

## Schema and Description

### `cluster_log.csv`

#### Description

Provides rich information on all jobs submitted to Slurm in each cluster.

#### Example
| job_id  | user  | vc    | gpu_num | cpu_num | node_num | state     | submit_time         | start_time          | end_time            | duration | queue |
| ------- | ----- | ----- | ------- | ------- | -------- | --------- | ------------------- | ------------------- | ------------------- | -------- | ----- |
| 1425511 | uXBbc | vcJkd | 1       | 1       | 1        | COMPLETED | 2020-06-09 18:41:01 | 2020-06-09 18:41:01 | 2020-06-10 04:55:09 | 36848    | 0     |
| 1425512 | uVMrF | vchbv | 4       | 16      | 1        | FAILED    | 2020-06-09 18:41:27 | 2020-06-09 18:41:27 | 2020-06-09 18:45:36 | 249      | 0     |
| 1425513 | uzqls | vcpDC | 1       | 1       | 1        | CANCELLED | 2020-06-09 18:41:28 | 2020-06-09 18:41:28 | 2020-06-17 14:15:21 | 675233   | 0     |

#### Schema

| Field         | Description                                         |
| ------------- | --------------------------------------------------- |
| `job_id`      | unique id of the job <sup>1</sup>                   |
| `user`        | hashed id for the user, prefix is '*u*'             |
| `vc`          | hashed id for the virtual cluster, prefix is '*vc*' |
| `gpu_num`     | number of GPUs required for the job                 |
| `cpu_num`     | number of CPUs required for the job                 |
| `node_num`    | number of nodes in the job                          |
| `state`       | the job's status upon termination  <sup>2</sup>     |
| `submit_time` | the job's submission time                           |
| `start_time`  | the job's start execution time                      |
| `end_time`    | the job's termination time                          |
| `duration`    | total job execution time of the job <sup>3</sup>    |
| `queue`       | total job queue time of the job <sup>4</sup>        |


#### Notes
1. `job_id` is generated by Slurm and reflects the job submission order in each cluster.
2. A job can end up with one of five statuses: (1) `COMPLETED`: it is finished successfully; (2) `CANCELLED`: it is terminated by the user; (3) `FAILED`: it is terminated due to internal or external errors; (4) `TIMEOUT`: the execution time is out of limit; (5) `NODE_FAIL`: it is terminated due to the node crash. `TIMEOUT` and `NODE_FAIL` are very rare in our traces, and are regarded as failed in our analysis. (Another status `SUSPENDED` happens only once in cluster *Uranus*, so we ignore it.)
3. Calculated from the difference between `end_time` and `start_time`. (Unit: seconds)
4. Calculated from the difference between `start_time` and `submit_time`. (Unit: seconds)



### ``cluster_gpu_number.csv``

#### Description

Lists the number of GPUs per day in each VC.

#### Example
| date       | vchbv | vc4om | vcVP5 | vc6YE | vchA3 | vccaA | vcTJs | vcvlY | vcSoL | vcMod | vcpDC | vc3sl | vc8Sj | vcJLV | vcLJZ | vcIya | vcJkd | vcdI0 | vcira | vcgkz | vcxS0 | vc7hD | vcXrB | vcvcM | vcp4O | total |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 2020-09-01 | 64    | 96    | 176   | 216   | 40    | 40    | 48    | 96    | 32    | 64    | 56    | 64    | 0     | 32    | 64    | 0     | 16    | 16    | 16    | 8     | 0     | 0     | 0     | 0     | 0     | 1144  |


#### Schema

| Field   | Description                             |
| ------- | --------------------------------------- |
| `date`  | record granularity is daily             |
| `vc***` | the number of GPUs of the VC            |
| `total` | the total number of GPUs of the cluster |
