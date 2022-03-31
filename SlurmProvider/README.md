These scripts are kept as back. The SlurmProvider is not working with the SSHChannel probably due to missing tunnels:

```
[Alvaro.Vidal@compute-0001 tmp]$ cat parsl.slurm.1648163466.0924313.submit.stdout
Found cores : 30
Launching worker: 1
Failed to find a viable address to connect to interchange. Exiting
```