class Mike:
    _client = None

    def __init__(self):
        self._client = 1
        Mike._client = 2


m = Mike()
print(m._client)
print(Mike._client)
