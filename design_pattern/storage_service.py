from abc import abstractmethod


class Singleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


class StorageService(Singleton):
    _client = None

    @property
    @abstractmethod
    def client(self):
        raise NotImplementedError

    @abstractmethod
    def upload_file(self):
        raise NotImplementedError


class GCloudStorageService(StorageService):
    @property
    def client(self):
        if not GCloudStorageService._client:
            GCloudStorageService._client = "GCloud client"
        return GCloudStorageService._client

    def upload_file(self):
        print(self.client + ": upload_file")


class MinioStorageService(StorageService):
    @property
    def client(self):
        if not MinioStorageService._client:
            MinioStorageService._client = "Minio client"
        return MinioStorageService._client

    def upload_file(self):
        print(self.client + ": upload_file")


if __name__ == "__main__":
    g1 = GCloudStorageService()
    g1.upload_file()
    g2 = GCloudStorageService()
    g2.upload_file()
    print(g1 is g2)

    m1 = MinioStorageService()
    m1.upload_file()
    print(m1 is g1)

    print(id(g1))
    print(id(g2))
    print(id(m1))
