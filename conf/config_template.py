import os

project_path = '/share/cp/projects/ajures/'
model_path = os.path.join(project_path, 'models')
meta_path = os.path.join(project_path, 'data')

track_path = '/share/cp/datasets/million_song_dataset/snippets/million_song_snippets_msdid/npy'
audio_path = '/share/cp/datasets/million_song_dataset/snippets/million_song_snippets_msdid/audio_only/'
musdb_storage = '/home/verena/data/cached_data/musdb_audios.pt'

web_folder = '/share/cp/temp/web/'
external_file_template = 'https://sanders.cp.jku.at/share.cgi/{}?ssid=07Tk2FX&fid=07Tk2FX&path=/{}&filename={}&openfolder=normal&ep='