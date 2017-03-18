def print_error(e, tag = ""):
    print "\033[1;31m[ERROR] " + str(e) + "\033[0;0m"

def print_warning(w, tag = ""):
    print "\033[1;35m[WARNING] " + str(w) + "\033[0;0m"
    pass

def print_msg(m, tag = ""):
    real_tag = "MESSAGE" if not tag else str(tag)
    print "[" + real_tag + "] " + str(m)
    pass
