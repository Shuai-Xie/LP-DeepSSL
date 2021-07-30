import paramiko


class SSH:
    def __init__(self, host, port, user, ssh_key_path, timeout=1800) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.ssh_key_path = ssh_key_path
        self.timeout = timeout

    def connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.host,
                       username=self.user,
                       port=self.port,
                       pkey=paramiko.RSAKey.from_private_key_file(self.ssh_key_path),
                       timeout=self.timeout)
        self.client = client

    def exec(self, cmd: str, timeout: int = 1800):
        stdin, stdout, stderr = self.client.exec_command(command=cmd, timeout=timeout)
        code = stdout.channel.recv_exit_status()
        if code == 0:
            return stdout.readlines()
        else:
            return stderr.readlines()

    def close(self):
        self.client.close()