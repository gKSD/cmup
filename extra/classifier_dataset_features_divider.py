import sys
import os
import numpy
import shutil

if len(sys.argv) < 2:
    raise Exception("Wrong argument: " + sys.argv[0] + " <feature_directory> [<train percent> <cv percent> <test percent>]")

directory = "calculated_essentia_features/all_dataset_for_counting/4_dataset_audio"
#directory = sys.argv[1]
print "DIRECTORY:\t" + directory

if not os.path.exists(directory):
    raise Exception("[ERROR] directory: '" + directory +"' doesn't exist!")

result_directory = "GROUPED"
train_dir = directory + "/" + result_directory + "/" + "train"
cv_dir    = directory + "/" + result_directory + "/" + "cv"
test_dir  = directory + "/" + result_directory + "/" + "test"
# TODO: check dir names if exist and modify or somehow

train_percent = int(sys.argv[2]) if len(sys.argv) == 3 else 60
cv_percent    = int(sys.argv[3]) if len(sys.argv) == 4 else 20
test_percent  = int(sys.argv[4]) if len(sys.argv) == 5 else 20

print "TRAIN PERCENT:\t" + str(train_percent) + "%"
print "CV PERCENT:\t" + str(cv_percent) + "%"
print "TEST PERCENT:\t" + str(test_percent) + "%"

label_dirs = os.listdir(directory)

os.makedirs(train_dir)
os.makedirs(cv_dir)
os.makedirs(test_dir)

for label_dir in label_dirs:
    os.makedirs(train_dir + "/" + label_dir)
    os.makedirs(cv_dir + "/" + label_dir)
    os.makedirs(test_dir + "/" + label_dir)

    files = numpy.array([])
    for f in os.listdir(directory + "/" + label_dir):
        if f.endswith(".json"):
            files = numpy.append(files, f)

    print "LABEL:\t" + label_dir + "\tFILES:\t" + str(files.size)

    train_count = int( files.size * (train_percent / 100.0) )
    cv_count    = int( files.size * (cv_percent    / 100.0) )
    test_count  = int( files.size * (test_percent  / 100.0) )

    print "TRAIN COUNT:\t" + str(train_count)
    print "CV COUNT:\t" + str(cv_count)
    print "TEST COUNT:\t" + str(test_count)

    base = directory + "/" + label_dir + "/"
    for i in range(0, train_count):
        shutil.copy(base + files[i], train_dir + "/" + label_dir + "/")
    for i in range(train_count, train_count + cv_count):
        shutil.copy(base + files[i], cv_dir + "/" + label_dir + "/")
    for i in range(train_count + cv_count, train_count + cv_count + test_count):
        shutil.copy(base + files[i], test_dir + "/" + label_dir + "/")

print "OK done"
