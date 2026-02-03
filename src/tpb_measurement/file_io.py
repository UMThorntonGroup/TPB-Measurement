import os
import pickle


class FileIO:
    base_dir = os.getcwd()

    def __init__(self, base_dir: str = None):
        if base_dir is not None:
            self.base_dir = base_dir
        self.create_output_directory()

    def create_output_directory(self):
        output_dir = os.path.join(self.base_dir, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def create_directory(self, directory_name: str):
        directory_path = os.path.join(self.base_dir, directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def create_pickle_dump(self, data, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle_dump(self, filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)
