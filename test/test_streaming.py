from src.file.FeatherFileReader import FeatherFileReader
from src.file.FeatherStreamReader import FeatherStreamReader
from src.file.FeatherStreamWriter import FeatherStreamWriter


if __name__ == '__main__':
    filename = 'data/test.feather'
    #filename = 'D:/Video/Cooperative_digging/segmentation/2024-08-29_16-11-00_SV11.predictions.feather'
    #filename = 'D:/Video/Cooperative_digging/tracked.feather'

    data = {
     'a': [1, 2, 3, 4],
     'b': ['foo', 'bar', 'baz', None],
     'c': [True, None, False, True]
    }

    stream_writer = FeatherStreamWriter(filename)
    stream_writer.write({'a': 1, 'b': 'foo', 'c': True})
    stream_writer.write({'a': 2, 'b': 'x', 'c': False})
    stream_writer.close()

    stream_reader = FeatherStreamReader([filename])
    #stream_reader = FeatherFileReader([filename])
    print(stream_reader.schema)
    for data in stream_reader.get_stream_iterator():
        print(data)
