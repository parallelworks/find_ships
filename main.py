import sys, os, json, time, glob
from random import randint
import argparse

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.providers import SlurmProvider, LocalProvider
from parsl.executors import HighThroughputExecutor

from parsl.addresses import address_by_hostname
from parsl.monitoring.monitoring import MonitoringHub

import pandas as pd

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

def json2dict(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def dex_dict_row(img_path, img_out_path):
    dict_row = json2dict(img_out_path.replace('png', 'json'))
    dict_row['img:original'] = img_path
    dict_row['img:processed'] = img_out_path
    return dict_row

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)

@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['cpu_executor']['HOST_IP'])
@bash_app(executors=['cpu_executor'])
def find_ships(run_dir, path_to_sing, find_script, img_path, model_dir, img_path_out, slurm_info = {},
            inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):

    if not slurm_info:
        slurm_info = {
            'nodes': '1',
            'partition': 'compute',
            'ntasks_per_node': '1',
            'walltime': '01:00:00'
        }

    return '''
        cd {run_dir}
        srun --nodes={nodes}-{nodes} --partition={partition} --ntasks-per-node={ntasks_per_node} --time={walltime} --exclusive singularity exec -B `pwd`:`pwd` -B {run_dir}:{run_dir} {path_to_sing} /usr/local/bin/python {find_script} \
            --img_path {img_path} \
            --model_dir {model_dir} \
            --img_path_out {img_path_out} \
    '''.format(
        run_dir = run_dir,
        path_to_sing = path_to_sing,
        find_script = find_script,
        img_path = img_path,
        model_dir = model_dir,
        img_path_out = img_path_out,
        nodes = slurm_info['nodes'],
        partition = slurm_info['partition'],
        ntasks_per_node = slurm_info['ntasks_per_node'],
        walltime = slurm_info['walltime']

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
                cores_per_worker = float(exec_conf['cpu_executor']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['cpu_executor']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider =  LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}'.format(
                        conda_sh = os.path.join(exec_conf['cpu_executor']['CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['cpu_executor']['CONDA_ENV'],
                        run_dir = exec_conf['cpu_executor']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['cpu_executor']['HOST_IP'],
                        username = os.environ['PW_USER'],
                        script_dir = exec_conf['cpu_executor']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                        key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                    )
                )
            )
        ],
        monitoring = MonitoringHub(
           hub_address = address_by_hostname(),
           resource_monitoring_interval = 5,
       ),
    )

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    # Make directory for output images:
    scheme_imgdir_out = args['imgdir_out'].split(':')[0]
    if scheme_imgdir_out != 'pw':
        raise(Exception('Only pw scheme is supported for the imgdir_out parameter!'))
    imgdir_out = args['imgdir_out'].split(':')[1].format(cwd = os.getcwd())
    os.makedirs(imgdir_out, exist_ok = True)

    # Find ships in image:
    scheme_imgdir = args['imgdir'].split(':')[0]
    if scheme_imgdir != 'pw':
        raise(Exception('Only pw scheme is supported for the imgdir parameter!'))
    imgdir = args['imgdir'].split(':')[1]

    find_ships_futs = []
    img_paths = glob.glob(os.path.join(imgdir, '*.png'))[0:1]
    img_paths_out = [ os.path.join(imgdir_out, os.path.basename(img_path)) for img_path in img_paths ]

    for img_path, img_path_out in zip(img_paths, img_paths_out):
        task_id = os.path.basename(img_path).split('.')[0]

        print('Finding ships in ' + img_path + ' --> ' + img_path_out, flush = True)

        find_ships_fut = find_ships(
            exec_conf['cpu_executor']['RUN_DIR'],
            exec_conf['cpu_executor']['SINGULARITY_CONTAINER_PATH'],
            './find_ships.py',
            './{}-in.png'.format(task_id),
            './model_dir',
            './{}-out.png'.format(task_id),
            slurm_info = {
                'nodes': args['nodes'],
                'partition': args['partition'],
                'ntasks_per_node': args['ntasks_per_node'],
                'walltime': args['walltime']
            },
            inputs_dict = {
                "find_script": {
                    "type": "file",
                    "global_path": "pw://{cwd}/find_ships/find_ships.py",
                    "worker_path": "{remote_dir}/find_ships.py".format(remote_dir =  exec_conf['cpu_executor']['RUN_DIR'])
                },
                "img_path": {
                    "type": "file",
                    "global_path": scheme_imgdir + "://" + img_path,
                    "worker_path": "{remote_dir}/{task_id}-in.png".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR'],
                        task_id = task_id
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
                    "worker_path": "{remote_dir}/{task_id}-out.png".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR'],
                        task_id = task_id
                    )
                },
                "stats_path_out": {
                    "type": "file",
                    "global_path": scheme_imgdir + "://" + img_path_out.replace('png', 'json'),
                    "worker_path": "{remote_dir}/{task_id}-out.json".format(
                        remote_dir = exec_conf['cpu_executor']['RUN_DIR'],
                        task_id = task_id
                    )
                }
            },
            stdout = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std-{}.out'.format(task_id)),
            stderr = os.path.join(exec_conf['cpu_executor']['RUN_DIR'], 'std-{}.err'.format(task_id))
        )

        find_ships_futs.append(find_ships_fut)


    for fut in find_ships_futs:
        fut.result()


    # Prepare design explorer files:
    print('\nCreating Design Explorer files', flush = True)
    dex_csv = os.path.join(os.getcwd(), 'dex.csv')
    dex_html = dex_csv.replace('csv', 'html')

    print('Creating CSV file ' + dex_csv, flush = True)
    dex_df = pd.DataFrame(
        [
            dex_dict_row(img, img_out) for img, img_out in zip(img_paths, img_paths_out)
        ]
    )
    dex_df.to_csv(dex_csv, index = False)

    print('Creating HTML file ' + dex_html, flush = True)
    dex_html = open(dex_html, 'w')
    dex_html.write(
        '''
        <html style="overflow-y:hidden;background:white"><a
            style="font-family:sans-serif;z-index:1000;position:absolute;top:15px;right:0px;margin-right:20px;font-style:italic;font-size:10px"
            href="/preview/DesignExplorer/index.html?datafile={csv}&colorby=num_ships"
            target="_blank">Open in New Window
        </a><iframe
            width="100%"
            height="100%"
            src="/preview/DesignExplorer/index.html?datafile={csv}&colorby=num_ships"
            frameborder="0">
        </iframe></html>
        '''.format(csv = dex_csv)
    )
    dex_html.close()
