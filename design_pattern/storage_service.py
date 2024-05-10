from abc import abstractmethod
import uuid


class Singleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


class StorageService(Singleton):
    pass


def client_generator():
    return uuid.uuid4()


class GCloudStorageService(StorageService):
    _client = None

    def set_client(value):
        GCloudStorageService._client = value

    def get_client(cls):
        return GCloudStorageService._client


class MinioStorageService(StorageService):
    _client = None

    def get_client(cls):
        return MinioStorageService._client


if __name__ == "__main__":
    GCloudStorageService.set_client(100)
    MinioStorageService._client = 200
    print(id(GCloudStorageService._client))
    print(id(MinioStorageService._client))
