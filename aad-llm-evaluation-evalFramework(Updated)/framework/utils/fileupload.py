import os
from openai import OpenAI
from typing import Optional
from datetime import datetime

class FileUpload:
    @staticmethod
    def get_client(api_key: Optional[str] = None):
        if api_key:
            return OpenAI(api_key=api_key)
        return OpenAI()  
 
    def __init__(self, api_key=None):
        self.client = self.get_client(api_key)
        self.file_path = None

    def upload(self, file_path: str):
        if os.path.exists(file_path):
            pass
        else:
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if file_path.endswith(".jsonl"):
            pass
        else:
            raise ValueError("The file must be a .jsonl file")


        with open(file_path, "rb") as f:
            uploaded = self.client.files.create(
                file=f,
                purpose="evals"
                )

        return uploaded.id

    def filepath(self, file_location: str):
        return self.upload(file_location)
    
    def get_created_timestamp(self, file_id: str):
        file_obj = self.client.files.retrieve(file_id)
        return file_obj.created_at
    
    def get_created_datetime(self, file_id: str):
        file_obj = self.client.files.retrieve(file_id)
        return datetime.fromtimestamp(file_obj.created_at).strftime("%Y-%m-%d %H:%M:%S")
    
    def filename(self,file_id: str):
        file_obj=self.client.files.retrieve(file_id)
        return file_obj.filename

    def status(self, file_id: str):
        file_obj = self.client.files.retrieve(file_id)
        return file_obj.status
        
    def list_files(self):
        file_list = self.client.files.list()
        return file_list.data  


