import os
import cv2
import jpegio as jpio
import pandas as pd
import sklearn.utils

class Alaska:
    def __init__(self, path, train_val_ratio, format="binary", multiclass_file=None):
        print("Reading data from {}".format(path))
        self.path = path
        self.format = format
        self.train_val_ratio = train_val_ratio
        if format == "binary":
            print("Preparing dataset in Binary Mode")
        elif format == "multiclass":
            print("Preparing dataset in MultiClass Mode")
        else:
            print("Format {} not supported.".format(format))

        self.JMiPOD_dict = {95: 1, 90: 2, 75: 3}
        self.JUNIWARD_dict = {95: 4, 90: 5, 75: 6}
        self.UERD_dict = {95: 7, 90: 8, 75: 9}
        self.multiclass_file = multiclass_file
        self.num_classes = None

    def build(self):
        ## 1. List images IDS
        # Cover Images
        cover_ids = os.listdir(os.path.join(self.path, 'Cover'))
        cover_ids = cover_ids
        for i in range(len(cover_ids)):
            cover_ids[i] = os.path.join(os.path.join(self.path, 'Cover'), cover_ids[i])

        # Crypted Images
        JMiPOD_ids = os.listdir(os.path.join(self.path, 'JMiPOD'))
        for i in range(len(JMiPOD_ids)):
            JMiPOD_ids[i] = os.path.join(os.path.join(self.path, 'JMiPOD'), JMiPOD_ids[i])

        JUNIWARD_ids = os.listdir(os.path.join(self.path, 'JUNIWARD'))
        for i in range(len(JUNIWARD_ids)):
            JUNIWARD_ids[i] = os.path.join(os.path.join(self.path, 'JUNIWARD'), JUNIWARD_ids[i])

        UERD_ids = os.listdir(os.path.join(self.path, 'UERD'))
        for i in range(len(UERD_ids)):
            UERD_ids[i] = os.path.join(os.path.join(self.path, 'UERD'), UERD_ids[i])

        self.JMiPOD_ids = JMiPOD_ids
        self.JUNIWARD_ids = JUNIWARD_ids
        self.UERD_ids = UERD_ids

        self.cover_ids = cover_ids

        ## 2. Set images id and labels by format
        if self.format == "binary":
            # This is a naive binary approach. The images that are modified are labeled as 1 while the orignal ones
            # are labeled as -1.
            crypt_labels = [1] * (len(JMiPOD_ids) + len(JUNIWARD_ids) + len(UERD_ids))
            cover_labels = [-1] * len(cover_ids)
            crypt_ids = JMiPOD_ids + JUNIWARD_ids + UERD_ids
            print("Number of images:"
                  "\n\t Cover: {} \n\t JMiPOD: {} \n\t JUNIWARD: {} \n\t UERD: {}".format(len(cover_ids),
                                                                                          len(JMiPOD_ids),
                                                                                          len(JUNIWARD_ids),
                                                                                          len(UERD_ids)))

            n_samples = int(len(cover_labels) * self.train_val_ratio)
            n_samples_val = len(cover_labels) - n_samples

            print("Splitting the dataset:\n\t - Training: \n\t\t Cover: {} \n\t\t Crypt {} \n\t - Validation: "
                  "\n\t\t Cover: {} \n\t\t Crypt {}".format(n_samples, n_samples * 3, n_samples_val, n_samples_val * 3))

            cover_ids = sklearn.utils.shuffle(cover_ids)
            crypt_ids = sklearn.utils.shuffle(crypt_ids)
            IMAGE_IDS_train = cover_ids[:n_samples] + crypt_ids[:n_samples * 3]
            IMAGE_LABELS_train = cover_labels[:n_samples] + crypt_labels[:n_samples * 3]

            IMAGE_IDS_val = cover_ids[-n_samples_val:] + crypt_ids[-n_samples_val * 3:]
            IMAGE_LABELS_val = cover_labels[-n_samples_val:] + crypt_labels[-n_samples_val * 3:]
            self.num_classes = 1
        elif self.format == "multiclass":
            # This is a multiclass approach. In here we consider each algorithm and each JPEG compression a class.
            # So in total we will have 10 classes
            JMiPOD_labels, JUNIWARD_labels, UERD_labels  = [], [], []

            cover_ids = sklearn.utils.shuffle(cover_ids)
            cover_labels = [0] * len(cover_ids)

            n_samples = int(len(cover_labels) * self.train_val_ratio)
            n_samples_val = len(cover_labels) - n_samples

            IMAGE_IDS_train = cover_ids[:n_samples]
            IMAGE_LABELS_train = cover_labels[:n_samples]

            IMAGE_IDS_val = cover_ids[-n_samples_val:]
            IMAGE_LABELS_val = cover_labels[-n_samples_val:]

            if self.multiclass_file is None:

                # Calculate labels for multiclass
                print("JMIPOD")
                for JMiPOD in JMiPOD_ids:
                    JMiPOD_labels.append(self.JMiPOD_dict[self.calculate_qf(JMiPOD)])
                print("juniward")
                for JUNIWARD in JMiPOD_ids:
                    JUNIWARD_labels.append(self.JUNIWARD_dict[self.calculate_qf(JUNIWARD)])
                print("uerd")
                for UERD in JMiPOD_ids:
                    UERD_labels.append(self.UERD_dict[self.calculate_qf(UERD)])

                crypt_ids = self.JMiPOD_ids + self.JUNIWARD_ids + self.UERD_ids
                crypt_labels = JMiPOD_labels + JUNIWARD_labels + UERD_labels

                df = pd.DataFrame(list(zip(crypt_ids, crypt_labels)), columns=['ids', 'label'])
                df.to_csv('./multiclass_stega_df.csv')
                exit()
            else:
                # Load file
                df = pd.read_csv(self.multiclass_file)

            labels_class = df['label'].unique().tolist()
            for label in labels_class:
                class_df = df.loc[df['label'] == label]
                path_ids = []
                for index, row in class_df.iterrows():
                    path_ids.append(row['ids'])
                path_ids = sklearn.utils.shuffle(path_ids)
                print("label {} with len {}".format(label,len(path_ids)))
                n_samples = int(len(path_ids) * self.train_val_ratio)
                n_samples_val = len(path_ids) - n_samples

                IMAGE_IDS_train += path_ids[:n_samples]
                IMAGE_LABELS_train += [label] * n_samples
                IMAGE_IDS_val += path_ids[-n_samples_val:]
                IMAGE_LABELS_val += [label] * n_samples_val
            self.num_classes = 1 + len(labels_class)

        return IMAGE_IDS_train, IMAGE_LABELS_train, IMAGE_IDS_val, IMAGE_LABELS_val

    def calculate_qf(self, image):
        jpegStruct = jpio.read(image)
        if (jpegStruct.quant_tables[0][0,0]==2):
            return 95
        elif (jpegStruct.quant_tables[0][0,0]==3):
            return 90
        elif (jpegStruct.quant_tables[0][0,0]==8):
            return 75
