import requests, os, click
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def get_lines(x):
    with open(x) as f:
        return f.readlines()
    
def download(x, path):
    ok = 0
    r = requests.get(x)
    fn = x.split('/')[-1]
    
    if r.status_code == requests.codes.ok:
        with open(path / fn, 'wb') as f:
            f.write(r.content)
        ok = 1
    
    r.close()
    return ok

@click.command()
@click.option('--path', '-p', prompt='Folder with txt files with urls.')
def main(path):
    path = Path(path)
    urls_files = [f for f in path.iterdir() if f.suffix == '.txt']
    urls = {j.strip() for i in map(get_lines, urls_files) for j in i if len(j.split('.')[-1]) in [3,4]}
    print(f'Valid urls: {len(urls)}')
    urls = [i for i in urls if i.split('/')[-1] not in os.listdir(path)]
    print(f'Non repeated urls: {len(urls)}')

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(partial(download, path=path), urls))
        
    print(f'Downloaded images: {sum(res)}')

if __name__ == '__main__':
    main()