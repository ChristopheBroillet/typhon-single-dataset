import datetime
import torch


def print_time(str):
    print()
    print(str, '--', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def print_results(metrics_dict, experiment):
    print(f"""
          RESULTS OF {experiment} ON THE TEST SET
          ---------------------------------------------------------------""")
    for metric_name, value in metrics_dict.items():
        print(f"""          {metric_name}: {value}""")
    print(f"""          ---------------------------------------------------------------""")
    print()


def print_gpu_memory(device):
    # print(torch.cuda.memory_summary(device=device))
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
