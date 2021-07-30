import sys
sys.path.insert(0, '/export/nfs/xs/codes/lp-deepssl')

import datetime
import os
import time
from queue import Queue
from threading import Thread
from utils.ssh import SSH


def get_remain_memory(mem_line):
    vals = mem_line.split()
    res = []
    for v in vals:
        if v.endswith('MiB'):
            res.append(int(v[:-3]))
            if len(res) == 2:
                break
    return res[1] - res[0]


def parse_gpu_resource(lines: list):
    gpus = {}
    for i in range(len(lines) - 1):
        if '%' in lines[i + 1]:  # unique mark
            gpu_id = lines[i].split()[1]
            gpus[gpu_id] = get_remain_memory(lines[i + 1])
    return gpus


class MMachine:
    def __init__(self,
                 hosts: list,
                 user: str,
                 ssh_key_path: str,
                 port: int = 22,
                 timeout: int = 1800):
        self.machines = []
        for host in hosts:
            ssh = SSH(host, port, user, ssh_key_path, timeout)
            self.machines.append(ssh)
        self.logs = Queue()  # better for multiple threads than list
        self.machine_number = len(hosts)

    def connect(self):
        for ssh in self.machines:
            ssh.connect()

    def close(self):
        for ssh in self.machines:
            ssh.close()

    def ssh_cmd(self, ssh, cmd: str, to_logs=False):
        ssh_lines = ssh.exec(cmd)
        if to_logs:
            self.logs.put([f'{ssh.host}\n{cmd}\n'] + ssh_lines)
        else:
            print(ssh.host)
            print(cmd)
            for line in ssh_lines:
                print(line, end='')

    def exec(self, cmd: str, multi_threads=False):
        if not multi_threads:
            for ssh in self.machines:
                self.ssh_cmd(ssh, cmd)
        else:
            self.logs.queue.clear()
            for ssh in self.machines:
                th = Thread(target=self.ssh_cmd, args=[ssh, cmd, True])
                th.start()
            cnt = 0
            while True:
                while self.logs.empty():
                    time.sleep(0.2)
                    print('#', end='', flush=True)
                print()
                log = self.logs.get()
                for line in log:
                    print(line, end='')
                print()
                cnt += 1
                if cnt == self.machine_number:
                    break

    def list_gpu_resource(self):
        for ssh in self.machines:
            ssh_lines = ssh.exec('nvidia-smi')
            print(ssh.host)
            for line in ssh_lines:
                print(line, end='')

    def exec_gpu(self,
                 cmd: str,
                 gpu_request: int = 10000,
                 max_workers: int = 10000,
                 write_logs: bool = False):
        num_workers = 0
        for ssh in self.machines:
            host = ssh.host
            gpus = parse_gpu_resource(lines=ssh.exec('nvidia-smi'))

            for gpu_id, memory in gpus.items():
                if memory >= gpu_request:
                    print(f'{host} gpu {gpu_id} remains {memory} MiB, can use!')
                    returns = ssh.exec(cmd.replace('--gpu_id 0',
                                                   f'--gpu_id {gpu_id}'))  # change gpu id
                    if write_logs:
                        filepath = f"logs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{host}_{gpu_id}.log"
                        with open(filepath, 'w') as f:
                            f.writelines(returns)
                    else:
                        for r in returns:
                            print(r, end='')

                    num_workers += 1
                    if num_workers == max_workers:
                        break
            if num_workers == max_workers:
                break
        print('total workers:', num_workers)

    def exec_gpu_nohup(self,
                       cmd: str,
                       envs: str = '',
                       conda_env: str = 'base',
                       gpu_request: int = 10000,
                       max_workers: int = 10000,
                       log_dir: str = 'logs',
                       identifier: str = ''):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        num_workers = 0
        for ssh in self.machines:
            host = ssh.host
            gpus = parse_gpu_resource(lines=ssh.exec('nvidia-smi'))

            for gpu_id, memory in gpus.items():
                if memory >= gpu_request:
                    log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{identifier}{num_workers}_{host}_{gpu_id}.log"
                    log_path = os.path.join(log_dir, log_name)

                    gpu_cmd = cmd.replace('--gpu_id 0', f'--gpu_id {gpu_id}')
                    nohup_cmd = f'{envs} nohup {gpu_cmd} > {log_path} 2>&1 &'
                    ssh.exec(f'conda activate {conda_env}; {nohup_cmd}')

                    print(f'{host} gpu {gpu_id} remains {memory} MiB, can use! log: {log_name}')

                    num_workers += 1
                    if num_workers == max_workers:
                        break

            if num_workers == max_workers:
                break

        print('total workers:', num_workers)


if __name__ == '__main__':
    mm = MMachine(
        hosts=[
            '10.252.192.38', '10.252.192.42', '10.252.192.43', '10.252.192.47', '10.252.192.48',
            '10.252.192.49'
        ],
        user='xs',
        ssh_key_path='/export/nfs/xs/.ssh/id_rsa',
    )
    mm.connect()

    mm.list_gpu_resource()

    # mm.exec(cmd='nvidia-smi', multi_threads=True)
    # mm.exec(cmd='docker system prune -f;docker images', multi_threads=True)

    # mm.exec(cmd='docker rmi dockerhub.jd.com/ime/framework:0.1', multi_threads=True)
    # mm.exec(cmd='docker pull dockerhub.jd.com/ime/framework:0.1', multi_threads=True)

    # mm.exec(cmd='docker rmi dockerhub.jd.com/ime/kubekit:0.2', multi_threads=True)
    # mm.exec(cmd='docker pull dockerhub.jd.com/ime/kubekit:0.2', multi_threads=True)

    # mm.exec(cmd='docker images | grep ime/', multi_threads=True)
    # mm.exec(cmd='docker images | grep torch/', multi_threads=True)

    mm.close()