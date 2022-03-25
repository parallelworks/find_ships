import sys, os, json, time, glob
from random import randint
import argparse

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor

import parsl_utils

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)

@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['cpu_executor']['HOST_IP'])
@bash_app(executors=['cpu_executor'])
def find_ships(run_dir, path_to_sing, find_script, img_path, model_dir, img_path_out,
            inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    return '''
        cd {run_dir}
        singularity exec -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {find_script} \
            --img_path {img_path} \
            --model_dir {model_dir} \
            --img_path_out {img_path_out} \
    '''.format(
        run_dir = run_dir,
        path_to_sing = path_to_sing,
        find_script = find_script,
        img_path = img_path,
        model_dir = model_dir,
        img_path_out = img_path_out
    )



if __name__ == '__main__':
    args = read_args()

    # Add sandbox directory
    for exec_label, exec_conf_i in exec_conf.items():
        if 'RUN_DIR' in exec_conf_i:
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(exec_conf_i['RUN_DIR'], 'run-' + str(randint(0, 99999)).zfill(5))
        else:
            base_dir = '/contrib/{PW_USER}/tmp'.format(PW_USER = os.environ['PW_USER'])
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(base_dir, 'run-' + str(randint(0, 99999)).zfill(5))

    config = Config(
        executors = [
            HighThroughputExecutor(
                worker_ports = ((int(exec_conf['cpu_executor']['WORKER_PORT_1']), int(exec_conf['cpu_executor']['WORKER_PORT_2']))),
                label = 'cpu_executor',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = int(exec_conf['cpu_executor']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['cpu_executor']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider =  SlurmProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['cpu_executor']['REMOTE_CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['cpu_executor']['REMOTE_CONDA_ENV'],
                        run_dir = exec_conf['cpu_executor']['RUN_DIR']
                    ),
                    #========GPU RUNS=============
                    #scheduler_options = '#SBATCH --gres=gpu:1', # For GPU runs!
                    #========CPU RUNS============
                    #scheduler_options = '#SBATCH --ntasks-per-node=40',  # DO NOT USE! Conflicts with cores_per_worker where Parsl sets --ntasks-per-node on separate SBATCH command, see note above.
                    partition =  exec_conf['cpu_executor']['PARTITION'],
                    nodes_per_block = int(exec_conf['cpu_executor']['NODES_PER_BLOCK']),
                    cores_per_node = int(exec_conf['cpu_executor']['CORES_PER_NODE']),   # Corresponds to --cpus-per-task
                    min_blocks = int(exec_conf['cpu_executor']['MIN_BLOCKS']),
                    max_blocks = int(exec_conf['cpu_executor']['MAX_BLOCKS']),
                    parallelism = 1,           # Was 0.80, 1 is "use everything you can NOW"
                    exclusive = False,         # Default is T, hard to get workers on shared cluster
                    walltime = exec_conf['cpu_executor']['WALLTIME'],     # Will limit job to this run time, 10 min default Parsl
                    channel = SSHChannel(
                        hostname = exec_conf['cpu_executor']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['cpu_executor']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                        key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                    )
                )
            )
        ]
    )

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    # Make directory for output images:
    scheme_imgdir_out = args['imgdir_out'].split(':')[0]
    if scheme_imgdir_out != 'pw':
        raise(Exception('Only pw scheme is supported for the imgdir_out parameter!'))
    imgdir_out = args['imgdir_out'].split(':')[1]
    os.makedirs(imgdir_out.format(cwd = os.getcwd()), exist_ok = True)

    # Find ships in image:
    scheme_imgdir = args['imgdir'].split(':')[0]
    if scheme_imgdir != 'pw':
        raise(Exception('Only pw scheme is supported for the imgdir parameter!'))
    imgdir = args['imgdir'].split(':')[1]

    find_ships_futs = []
    for img_path in glob.glob(os.path.join(imgdir, '*.png')):
        img_path_out = os.path.join(imgdir_out, os.path.basename(img_path))
        task_id = os.path.basename(img_path).split('.')[0]

        print('Finding ships in ' + img_path + ' --> ' + img_path_out, flush = True)

        find_ships_fut = find_ships(
            exec_conf['cpu_executor']['RUN_DIR'],
            exec_conf['cpu_executor']['SINGULARITY_CONTAINER_PATH'],
            '{remote_dir}/find_ships.py'.format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR']),
            '{remote_dir}/img-in.png'.format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR']),
            '{remote_dir}/model_dir'.format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR']),
            '{remote_dir}/img-out.png'.format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR']),
            inputs_dict = {
                "find_script": {
                    "type": "file",
                    "global_path": "pw://{cwd}/find_ships/find_ships.py",
                    "worker_path": "{remote_dir}/find_ships.py".format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR'])
                },
                "img_path": {
                    "type": "file",
                    "global_path": scheme_imgdir + "://" + img_path,
                    "worker_path": "{remote_dir}/img-in.png".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR']
                    )
                },
                "model_dir": {
                    "type": "directory",
                    "global_path": args['model_dir'],
                    "worker_path": "{remote_dir}/model_dir".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR']
                    )
                },
            },
            outputs_dict = {
                "img_path_out": {
                    "type": "file",
                    "global_path": scheme_imgdir + "://" + img_path_out,
                    "worker_path": "{remote_dir}/img-out.png".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR']
                    )
                }
            },
            stdout = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std-{}.out'.format(task_id)),
            stderr = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std-{}.err'.format(task_id))
        )

        find_ships_futs.append(find_ships_fut)
        #
        break

    for fut in find_ships_futs:
        fut.result()
