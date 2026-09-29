import kagglehub

path = kagglehub.competition_download(
    'asap-aes',
    output_dir='./data/raw'
)

print("Downloaded to:", path)