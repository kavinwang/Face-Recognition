import codecs


def get_script_params(script_file):
    argv = []
    with codecs.open(script_file, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue
            lineArr = line.split(' ')
            argv.extend([lineArr[0], lineArr[1]])
    return argv


def test(module, script_file):
    module.main(module.parse_arguments(get_script_params(script_file)))
    print('complete test')
