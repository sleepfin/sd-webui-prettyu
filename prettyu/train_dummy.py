import time
import tqdm


def main():
    for i in tqdm.tqdm(range(1000)):
        time.sleep(0.01)


if __name__ == "__main__":
    main()
