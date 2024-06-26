import os
import urllib.request
import zipfile
import tqdm


class DataDownloader:
    def __init__(self):
        self.dataset_list = {
            'har70plus': 'https://www.archive.ics.uci.edu/static/public/780/har70.zip',
            'harth': 'https://archive.ics.uci.edu/static/public/779/harth.zip',
        }

    def get_dataset_list(self):
        '''
        Get the list of datasets available for download
        :return: List of datasets available for download
        '''
        return self.dataset_list

    def get_dataset_url(self, dataset_name):
        '''
        Get the URL for the given dataset
        :param dataset_name: The name of the dataset
        :return: The URL for the given dataset
        '''
        if dataset_name not in self.dataset_list:
            return None
        return self.dataset_list[dataset_name]

    def download_data(self, dataset_name, save_path='../data'):
        '''
        Download the given dataset and save it to the save_path
        :param dataset_name: The name of the dataset to download
        :param save_path: The path to save the dataset
        :return: Boolean value indicating if the download was successful
        '''
        url = self.get_dataset_url(dataset_name)
        if url is None:
            return False

        # Prepare the save_path
        zip_path = os.path.join(save_path, dataset_name)

        # Check if the save_path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Check if the zip_path exists
        if not os.path.exists(zip_path):
            os.makedirs(zip_path)

        # Check if the dataset is already downloaded
        if os.path.exists(zip_path):
            # Check if there are csv files in the directory
            if len([f for f in os.listdir(zip_path) if f.endswith('.csv')]) > 0:
                print('Dataset already downloaded')
                return True

        # Download the dataset with a progress bar
        print('Downloading dataset: ', dataset_name)

        # The progress bar should show the total size of the file
        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading " + dataset_name) as t:
            urllib.request.urlretrieve(url, os.path.join(save_path, dataset_name + '.zip'), reporthook=self.tqdm_hook(t))

        with zipfile.ZipFile(os.path.join(save_path, dataset_name + '.zip'), 'r') as zip_ref:
            with tqdm.tqdm(unit='files', total=len(zip_ref.namelist()), desc='Extracting files') as t:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, save_path)
                    t.update(1)

        # Delete the zip file
        os.remove(os.path.join(save_path, dataset_name + '.zip'))

        print('Download complete')

        return True

    def tqdm_hook(self, t):
        '''
        Function to update the progress bar for how much of the file has been downloaded
        :param t: The progress bar object
        :return: The hook function
        '''

        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            '''
            Update the progress bar
            :param b: The number of blocks downloaded
            :param bsize: The size of each block
            :param tsize: The total size of the file
            '''
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return update_to


if __name__ == '__main__':
    downloader = DataDownloader()
    success = downloader.download_data('har70plus')

    print('Download successful: ', success)
