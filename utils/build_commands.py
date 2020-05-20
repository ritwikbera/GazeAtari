import json

def build_commands(filename='dataset/action_enums.txt'):

    commands = {}
    with open(filename) as fh:
        for line in fh:
            if line[0] == '#' or line[0] == '\n':
                continue
            # print(line.strip().split('='))
            command, description = line.strip().split('=')
            commands[command.strip()] = description.strip()
            # print(commands)

    commands = {int(v): k for k, v in commands.items()}
    return commands 
    # return json.dumps(commands, indent=2, sort_keys=True)

if __name__=='__main__':
    print(build_commands(filename='/home/deathstar/work/GazeGuidedImitation/dataset/action_enums.txt'))
