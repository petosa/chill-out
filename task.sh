# The purpose of the script is to bulk-execute many calls to main.py so that you can automate a bunch of experiments without babysitting.

for n in 500 1000 1500 2000 
do
    echo $n
    tmp=$(mktemp)
    # The following line edits the config, make sure you have jq installed.
    (jq '.loader.args.train_size = '$n'' < config.json ) > "$tmp" && mv "$tmp" config.json 
    python main.py policy uf
    python main.py search
done
