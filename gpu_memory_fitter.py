import torch
import subprocess
import re
import logging
import sys
from math import floor, ceil

class GPUMemoryFitter:
    """
    This class is used to manage GPU memory allocation for large machine learning
    models. I need this capability due to sharing out my GPU workstation among
    friends and colleagues. It may seem assinine if you are a single user on a 
    single or even dual GPU system where you always know what resources are
    available and what resources are not. For me, it is necessary.
    
    This class rounds DOWN available memory and rounds UP memory needs.
    It is possible to end up in a scenario where you are fully capable of loading
    and running a model, but this fitter ends up refusing to generate a usable
    device_map for your hardware configuration.

    This class is semi-topology aware through the use of nvidia-smi. Therefore, 
    this class relies on the use of nvidia-smi. If you don't run nvidia cards,
    this code will not (likely?) work for you. I don't know how the necessary 
    values show up on non-nvidia tools, I'm happy to add other platform equiv if
    someone can show me what outputs or calls will work to get the same info.

    The topology awareness makes this fitter better in multi-user or mixed nvlink
    topologies. I've got 3 A6000 cards and two are connected via NVLink. I would
    not want to use an NVLink card if my model fits in a single non-NVLink card.
    
    Attributes:
        mem_per_billion_params (float): Memory required per billion model parameters.
            Note:
                4bit quantized, .65 is usually appropriate
                6bit quantized, .85 is usually appropriate
                8bit, 1.1 is usually safe
                16bit, 2.3 is usually safe
                32bit, 4.6 is usually safe
        available_mem_fudge_factor (float): Factor to adjust available memory estimations.
            Note:
                This is here to allow you to fiddle around with how much mem your hardware
                is calculated to have. These are here to reduce available memory to
                account for kernels and other usage that isn't explicitly the fault of the
                model itself.
                If you have nothing else running, 1 is usually safe.
    """
    
    def __init__(self, mem_per_billion_params=1.1, available_mem_fudge_factor=1):
        """
        Initializes the GPUMemoryFitter with specified parameters.

        Args:
            mem_per_billion_params (float): Memory required per billion model parameters.
            Note:
                4bit quantized, .65 is usually appropriate
                6bit quantized, .85 is usually appropriate
                8bit, 1.1 is usually safe
                16bit, 2.3 is usually safe
                32bit, 4.6 is usually safe
            available_mem_fudge_factor (float): Factor to adjust available memory estimations.
        """
        if mem_per_billion_params <= .1:
            self.mem_per_billion_params = 1.1
        else:
            self.mem_per_billion_params = mem_per_billion_params
        if available_mem_fudge_factor < 0:
            self.available_mem_fudge_factor = 1
        else: 
            self.available_mem_fudge_factor = available_mem_fudge_factor
        self._setup_logging()

    def _setup_logging(self):
        """
        Sets up logging for the class.
        """
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        log_format = '%(asctime)s - %(name)s - %(levelname)s - line: %(lineno)d - %(message)s'
        formatter = logging.Formatter(log_format)

        # Check if the desired handler is already added
        if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            root.addHandler(handler)

        self.log = logging.getLogger(__name__)

    def _get_gpu_nvlink_status(self):
        """
        Runs the nvidia-smi command to get NVLink information and identifies GPUs with NVLink 
        enabled or disabled.

        Returns:
            tuple: A tuple containing two lists, the first list contains IDs of GPUs with 
                   NVLink enabled, and the second list contains IDs of GPUs with NVLink 
                   disabled.
        """
        result = subprocess.run(['nvidia-smi', 'nvlink', '--status'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

        nvlink_enabled_gpus = set()
        nvlink_disabled_gpus = set()
        gpu_id = None

        for line in output.split('\n'):
            if 'GPU' in line:
                match = re.search(r"GPU (\d)", line)
                if match:
                    gpu_id = int(match.group(1))
                    nvlink_disabled_gpus.add(gpu_id)
            elif 'GB/s' in line and gpu_id is not None:
                nvlink_enabled_gpus.add(gpu_id)
                nvlink_disabled_gpus.discard(gpu_id)

        return list(nvlink_enabled_gpus), list(nvlink_disabled_gpus)

    def create_device_map(self, model_size_billion_params, mem_per_billion_params=-1):
        """
        Creates a map of devices to allocate memory for a model based on its size and 
        available GPU memory.

        Args:
            model_size_billion_params (float): Size of the model in billions of parameters.
            mem_per_billion_params (float): Size of memory per billions of parameters
            Note:
                4bit quantized, .65 is usually appropriate
                6bit quantized, .85 is usually appropriate
                8bit, 1.1 is usually safe
                16bit, 2.3 is usually safe
                32bit, 4.6 is usually safe
                This should include the model AND working memory.

        Returns:
            dict: A dictionary representing the allocation of memory to each GPU.

        Raises:
            RuntimeError: If there is insufficient GPU memory for the model.
        """
        if mem_per_billion_params < 0:
            mem_per_billion_params = self.mem_per_billion_params

        total_model_memory = ceil(model_size_billion_params * mem_per_billion_params)

        if not torch.cuda.is_available():
            import psutil
            available_memory = psutil.virtual_memory().available
            self.log.debug("CUDA is not available. Using CPU.")
            return {"cpu": floor((available_memory - self.available_mem_fudge_factor) / (2**30))}
        
        self.log.info("Evaluating GPU topology...")
        gpus_nvlink_enabled, gpus_nvlink_disabled = self._get_gpu_nvlink_status()
        gpus_all = list(gpus_nvlink_enabled + gpus_nvlink_disabled)
        if len(gpus_all) > torch.cuda.device_count():
            self.log.error(f"Mismatch in nvidia-smi ({len(gpus_all)}) and torch.cuda.device_count ({torch.cuda.device_count()})")
            gpus_nvlink_enabled = []
            gpus_nvlink_disabled = list(range(torch.cuda.device_count()))
            gpus_all = list(range(torch.cuda.device_count()))
        self.log.info(f"Total available GPUs: {len(gpus_all)} ({gpus_all})")
        self.log.info(f"GPUs with NVLink enabled: {gpus_nvlink_enabled}")
        self.log.info(f"GPUs with NO NVLink: {gpus_nvlink_disabled}")

        # Get available memory for each GPU
        gpu_memory = {}
        for gpu_id in gpus_all:
            gpu_id=int(gpu_id)
            available_memory = floor((torch.cuda.mem_get_info(gpu_id)[0] - self.available_mem_fudge_factor) / (2**30))  # Convert to GB
            available_memory = available_memory if available_memory >= 0 else 0
            gpu_memory[gpu_id] = available_memory
        self.log.info("GPU memory availability: " + str(gpu_memory))

        # Sort GPUs based on available memory
        sorted_gpus = sorted(gpu_memory, key=gpu_memory.get, reverse=True)
        gpu_memory_dict = {gpu_id: gpu_memory[gpu_id] for gpu_id in gpus_nvlink_enabled}
        sorted_nvlink_enabled = sorted(gpu_memory_dict, key=lambda x: gpu_memory_dict[x], reverse=False)
        gpu_memory_dict = {gpu_id: gpu_memory[gpu_id] for gpu_id in gpus_nvlink_disabled}
        sorted_nvlink_disabled = sorted(gpu_memory_dict, key=lambda x: gpu_memory_dict[x], reverse=True)
        gpu_memory_dict = None
        self.log.info("GPU memory order: " + str(sorted_gpus))
        self.log.info("Topology evaluation complete.")

        # Allocate model parts to GPUs in the order of sorted GPUs
        device_map = {}
        remaining_memory = total_model_memory
        self.log.info("Model needs " + str(remaining_memory) + "GB of memory." )

        # Check to see if the model fits entirely in a single card without NVLink
        for gpu_id in sorted_nvlink_disabled:
            available_memory = gpu_memory[gpu_id]
            if remaining_memory <= available_memory:
                self.log.info(f"GPU {gpu_id} Memory: {available_memory} GB\nModel fits entirely in a single non-NVLink card")
                device_map[gpu_id] = remaining_memory
                device_map["cpu"] = 0
                return device_map

        #Reset unsuccessful device_map
        remaining_memory = total_model_memory
        device_map = {}

        # Check to see if the model fits entirely within NVLink connected cards
        for gpu_id in sorted_nvlink_enabled:
            available_memory = gpu_memory[gpu_id]
            self.log.info(f"GPU {gpu_id} Memory: {available_memory} GB")
            if remaining_memory <= available_memory:
                device_map[gpu_id] = remaining_memory
                device_map["cpu"] = 0
                return device_map
            device_map[gpu_id] = available_memory
            remaining_memory -= available_memory
        
        #Reset unsuccessful device_map
        remaining_memory = total_model_memory
        device_map = {}

        # Check to see if the model fits within all cards combined
        for gpu_id in sorted_gpus:
            available_memory = gpu_memory[gpu_id]
            self.log.info(f"GPU {gpu_id} Memory: {available_memory} GB")
            if remaining_memory <= available_memory:
                device_map[gpu_id] = remaining_memory
                device_map["cpu"] = 0
                return device_map
            device_map[gpu_id] = available_memory
            remaining_memory -= available_memory

        # Make sure we fit the model in the cards
        if remaining_memory > 0:
            self.log.error("Insufficient GPU memory for the model, offloading to CPU and system RAM.")
            device_map["cpu"] = remaining_memory
            return device_map

        raise("Unexpected condition occurred. This line should never be reached.")
    def _to_percentage(self, arr):
        total = sum(arr)
        return [round((x / total), 2) for x in arr]

    def create_llama_cpp_tensor_split(self, model_size_billion_params, mem_per_billion_params=-1):
        """
        Creates a map of devices to allocate memory for a model based on its size and available GPU memory.
        Args:
            model_size_billion_params (int): Size of the model in billions of parameters.
            mem_per_billion_params (float): Size of memory per billions of parameters
            Note:
                4bit quantized, .65 is usually appropriate
                6bit quantized, .85 is usually appropriate
                8bit, 1.1 is usually safe
                16bit, 2.3 is usually safe
                32bit, 4.6 is usually safe
                This should include the model AND working memory.

        Returns:
            dict:
                "tensor": array: An array representing the proportions of memory per device (including CPU).
                "main_gpu": int: An integer representing the largest proportioned GPU from the tensor list.
        Raises:
            RuntimeError: If there is insufficient GPU memory for the model.
        """
        if mem_per_billion_params < 0:
            mem_per_billion_params = self.mem_per_billion_params
        total_model_memory = ceil(model_size_billion_params * mem_per_billion_params)

        device_map = self.create_device_map(model_size_billion_params, mem_per_billion_params)

        #handles case of no GPUs
        if not torch.cuda.is_available():
            import psutil
            available_memory = psutil.virtual_memory().available
            self.log.debug("CUDA is not available. Using CPU.")
            return [100]
        return_array = []
        array_values = []
        #handles case of one or more GPUs
        for gpu_id in range(torch.cuda.device_count()):
            if gpu_id in device_map.keys():
                array_values.append(device_map[gpu_id])
            else:
                array_values.append(0)
        array_values.append(device_map['cpu'])

        tensor_list = self._to_percentage(array_values)
        return {
                'tensor': tensor_list,
                'main_gpu': tensor_list.index(max(tensor_list[:-1]))
                }

# Example usage
if __name__ == "__main__":
    model_sizes = [1,2,3,4] + [*range(5,156,10)]
    gpu_manager = GPUMemoryFitter()
    for model_size in model_sizes:
        device_map = gpu_manager.create_device_map(model_size)
        print(f"Device map for the model with {model_size} billion parameters:", device_map)
