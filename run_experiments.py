import argparse
import ast
from multiprocessing import pool
import time
import subprocess


def run_cmd(cmd):
    print(f'Running {cmd}')
    finished = True
    try:
        a = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'The command {cmd} failed with code {e.returncode}:')
        print(f'{e.stderr}')
        finished = False
    return finished, cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cmd-file', action='store', nargs=1, type=str, required=True,
        help='Path to file containing the commands to be run in parallel. The file must contain a '
             'list of commands readable by ast.literal_eval().'
    )
    parser.add_argument(
        '--nthreads', action='store', nargs=1, type=int, required=True,
        help='Maximum number of threads that will be run in parallel.'
    )
    parser.add_argument(
        '--wait', action='store', nargs=1, type=int, required=False,
        help='The number of seconds to wait before starting this script.'
    )
    args = parser.parse_args()

    if args.wait is not None:
        print(f'Waiting {args.wait[0]} seconds before starting.')
        time.sleep(args.wait[0])

    with open(args.cmd_file[0]) as cmd_file:
        commands = ast.literal_eval(cmd_file.read())

    thread_pool = pool.Pool(processes=args.nthreads[0])

    results = thread_pool.map(run_cmd, commands)
    for res in results:
        if res[0]:
            print(f'Finished {res[1]}')
        else:
            print(f'Failed   {res[1]}')