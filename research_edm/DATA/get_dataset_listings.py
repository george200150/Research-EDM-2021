import os
import glob


def main():
    everything = glob.glob("datasets" + '**/**', recursive=True)
    print(["/".join(x.split(os.path.sep)) for x in everything if x.endswith(".txt")])


if __name__ == '__main__':
    main()
