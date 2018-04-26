# coding=utf-8
import tensorflow as tf


class BatchGener:
    def __init__(self, train_path, embedding_size, batch_size=1):
        self.train_path = train_path
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        self.data_index = 0

        self.data, self.length, self.station_num = self.pre_process()
        print("station_num:", self.station_num)
        print("record_num:", self.length)

        sess = tf.Session()
        embeddings = tf.Variable(tf.random_uniform([self.station_num, self.embedding_size], -1.0, 1.0))
        saver = tf.train.Saver([embeddings])
        #  词向量存储位置
        saver.restore(sess, "/home/jiang/DeepSpace/HumanTravelPredictioninHaiNan/data/station_embedding.ckpt")
        self.embed = embeddings.eval(session=sess)
        return

    def pre_process(self):
        """
        :return: [[station_record],[station_record]], record_nums
        """
        train_list = self.read_data()
        station_list, station_num = self.read_station()
        new_list = []
        length = len(train_list)
        station_indices = dict((c, i) for i, c in enumerate(station_list))
        indices_station = dict((i, c) for i, c in enumerate(station_list))
        # print station_indices
        for record in train_list:
            new_record = []
            for one_station in record:
                new_station = station_indices[one_station]
                new_record.append(new_station)
            new_list.append(new_record)
        return new_list, length, station_num

    def read_data(self):
        file_object = open(self.train_path)
        train_list = file_object.readlines()
        train_set = []
        user_record = []
        for i, record in enumerate(train_list):
            record = record.strip()
            user, station = (record.split(",")[j] for j in [0, 3])
            if i == len(train_list) - 1:
                user_record.append(int(station))
                train_set.append(user_record)
                return train_set
            elif user != train_list[i + 1].split(",")[0]:
                user_record.append(int(station))
                train_set.append(user_record)
                user_record = []
            else:
                user_record.append(int(station))

    def read_station(self):
        file_object = open(self.train_path + '_station')
        station_list = []
        stations = file_object.readlines()
        for station in stations:
            station = station.strip()
            station_list.append(int(station))
        return station_list, len(station_list)

    def generate_batch(self):
        """
        :return: X:word2vec Y:one-hot
        """
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            x_record = []
            y_record = []
            for j in range(len(self.data[self.data_index]) - 1):
                x_record.append(self.embed[self.data[self.data_index][j]])
                y_record.append(self.data[self.data_index][j + 1])
            x_batch.append(x_record)
            y_batch.append(y_record)
            self.data_index += 1
            if self.data_index == self.length:
                self.data_index = 0
        return x_batch, y_batch

    def two_vector_generate_batch(self):
        """
        :return: X:word2vec Y:word2vec
        """
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            x_record = []
            y_record = []
            for j in range(len(self.data[self.data_index]) - 1):
                x_record.append(self.embed[self.data[self.data_index][j]])
                y_record.append(self.embed[self.data[self.data_index][j + 1]])
            x_batch.append(x_record)
            y_batch.append(y_record)
            self.data_index += 1
            if self.data_index == self.length:
                self.data_index = 0
        return x_batch, y_batch


if __name__ == '__main__':
    B = BatchGener('/home/jiang/data/sorted10000', 100, 22)
    # for i in range(1):
    #     x_batch, y_batch = B.generate_batch()
    #     print x_batch, y_batch
