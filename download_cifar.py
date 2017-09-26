import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--chunk-size', type=int, default=5000, help='Chunk size in bytes')
parser.add_argument('-p', '--print-every', type=int, default=1, help='Print progress')

args = parser.parse_args()

response = requests.get('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', stream=True)
if response.ok:
    total_bytes = int(response.headers['Content-Length'])
    downloaded_bytes = 0
    with open('cifar10.tar.gz', 'wb') as fd:
        for i, chunk in enumerate(response.iter_content(args.chunk_size)):
            fd.write(chunk)
            downloaded_bytes += args.chunk_size
            downloaded_percents = downloaded_bytes / total_bytes * 100

            if i and i % args.print_every == 0:
                print('\rDownloaded {:.2f}%'.format(downloaded_percents), end='')
else:
    print('Response is not OK. Try restart download.')