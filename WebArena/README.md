# Running Memory Agent on WebArena

## WebArena Environment

First of all, you need to make sure that all WebArena websites are on. First of all check with:

```bash
docker ps
```

You may see some images are running. If not, go a set up directory to run these dockers.

```bash
cd /home/nus/zijie/webarena_setup/webarena-setup/webarena
sudo bash 04_docker_start_containers.sh
```

You may refer to [this website](https://github.com/gasse/webarena-setup/tree/main/webarena) for more information.

## Running Memory Agent

After setting up the WebArena environment, you can start running memory agent on it. Go to the project directory. You shall see a `env_setup.txt` under this directory, copy and run the commands in it. PLEASE REMINDE that this file contains API KEY, so do not push it to any git repo.

After that, you are able to run memory agent via:

```bash
python run.py --task_name webarena.0
```

You may assign `--task_name` to different values (For example, `webarena.102`) to run different webarena task.

To run multiple tasks for several times, you may run:

```
# This command runs WebArena test cases 0 to 5 for 3 times sequentially.
python run.py --test_case_start 0 --test_case_end 5 --repeat_runs 3
```


python test_webarena_lite.py --tasks 68 --repeat 10 --model google/gemini-2.5-flash-preview-09-2025 --llm_eval google/gemini-2.5-flash-preview-09-2025 --max_steps 25 --workers 1 --use_screenshot_action --use_screenshot_eval


nohup python test_webarena_lite.py --model google/gemini-2.5-flash-preview-09-2025 --llm_eval google/gemini-2.5-flash-preview-09-2025 --max_steps 25 --workers 8 --disable_memory 