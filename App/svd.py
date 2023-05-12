from modules.preprocessing import load_data
import torch as pt

if __name__ == "__main__":
    cp = load_data("cp_clean.pt")
    keys_084 = [key for key in cp.keys() if "0.84" in key]
    print(keys_084) 
    cp_084 = {key: cp[key] for key in keys_084}

    for i, key in enumerate(keys_084):
        print(cp_084[key].shape)
        cp_084[key] = cp_084[key][:,:,:500]
        print(cp_084[key].shape)

    pt.save(cp_084, "./data/cp_084_500snaps.pt")


