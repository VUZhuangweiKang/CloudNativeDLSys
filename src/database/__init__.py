class DLCacheCollection:
    Client = 0
    Job = 1
    Dataset = 2

class ChunkStatus:
    PREPARE = 0
    ACTIVE = 1
    PENDING = 2
    COOL_DOWN = 3
    INACTIVE = 4