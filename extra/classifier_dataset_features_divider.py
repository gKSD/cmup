import sys
import os
import numpy
import shutil

if len(sys.argv) < 2:
    raise Exception("Wrong argument: " + sys.argv[0] + " <feature_directory> train[=<percent>] test[=<percent>] cv[=<percent>]")

directory = "calculated_essentia_features/all_dataset_for_counting/4_dataset_audio"
#directory = sys.argv[1]
print "DIRECTORY:\t" + directory

if not os.path.exists(directory):
    raise Exception("[ERROR] directory: '" + directory +"' doesn't exist!")

need_cv       = False
need_test     = False
need_train    = False
train_percent = 0
cv_percent    = 0
test_percent  = 0
for i in sys.argv:
    if i[:2] == "cv":
        need_cv = True
        ar = i.split("=")
        if len(ar) == 2:
            cv_percent = int(ar[1])
        else:
            cv_percent = 20
    if i[:4] == "test":
        need_test = True
        ar = i.split("=")
        if len(ar) == 2:
            test_percent = int(ar[1])
        else:
            test_percent = 20
    if i[:5] == "train":
        need_train = True
        ar = i.split("=")
        if len(ar) == 2:
            train_percent = int(ar[1])
        else:
            train_percent = 60

if not(need_train) and not(need_cv) and not(need_test):
    print "[WARNING] print do not need to make any division: test, train or cv"
    sys.exit()

print "TRAIN PERCENT:\t" + str(train_percent) + "%"
print "CV PERCENT:\t" + str(cv_percent) + "%"
print "TEST PERCENT:\t" + str(test_percent) + "%"

if (test_percent + train_percent + cv_percent) > 100:
    raise Exception("Invalid percentage division for splits!")

result_directory = directory + "/" + "GROUPED"
train_dir = result_directory + "/" + "train"
cv_dir    = result_directory + "/" + "cv"
test_dir  = result_directory + "/" + "test"
# TODO: check dir names if exist and modify or somehow

label_dirs = os.listdir(directory)

os.makedirs(result_directory)

if need_train:
    os.makedirs(train_dir)
if need_cv:
    os.makedirs(cv_dir)
if need_test:
    os.makedirs(test_dir)


for label_dir in label_dirs:
    if need_train:
        os.makedirs(train_dir + "/" + label_dir)
    if need_cv:
        os.makedirs(cv_dir + "/" + label_dir)
    if need_test:
        os.makedirs(test_dir + "/" + label_dir)

    files = numpy.array([])
    for f in os.listdir(directory + "/" + label_dir):
        if f.endswith(".json"):
            files = numpy.append(files, f)

    print "LABEL:\t" + label_dir + "\tFILES:\t" + str(files.size)

    train_count = int( files.size * (train_percent / 100.0) )
    cv_count    = int( files.size * (cv_percent    / 100.0) )
    test_count  = int( files.size * (test_percent  / 100.0) )

    if need_train:
        print "TRAIN COUNT:\t" + str(train_count)
    if need_cv:
        print "CV COUNT:\t" + str(cv_count)
    if need_test:
        print "TEST COUNT:\t" + str(test_count)

    base = directory + "/" + label_dir + "/"
    for i in range(0, train_count):
        shutil.copy(base + files[i], train_dir + "/" + label_dir + "/")
    for i in range(train_count, train_count + cv_count):
        shutil.copy(base + files[i], cv_dir + "/" + label_dir + "/")
    for i in range(train_count + cv_count, train_count + cv_count + test_count):
        shutil.copy(base + files[i], test_dir + "/" + label_dir + "/")
        """
    for i in range(0, train_count):
        f = files[0]
        f = numpy.delete(files, 0)
        shutil.copy(base + f, train_dir + "/" + label_dir + "/")
    for i in range(0, cv_count):
        f = files[0]
        f = numpy.delete(files, 0)
        shutil.copy(base + f, cv_dir + "/" + label_dir + "/")
    for i in range(0, test_count):
        f = files[0]
        f = numpy.delete(files, 0)
        shutil.copy(base + f, test_dir + "/" + label_dir + "/")
        """

print "OK done"
