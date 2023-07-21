import multiprocessing
import subprocess

def run_script(cluster, sch):
    subprocess.call(["python3", "scheduler.py", "--cluster", cluster, "--sch", sch])

if __name__ == "__main__":
    processes = []
    
    clusters = ['earth', 'saturn', 'uranus', 'venus']
    schedulers = ['ours', 'bf', 'ff', 'wf', 'csa']

    for cluster in clusters:
        for sch in schedulers:
            p = multiprocessing.Process(target=run_script, args=(cluster, sch))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
