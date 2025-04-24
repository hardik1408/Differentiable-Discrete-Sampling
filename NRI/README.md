# NRI
- Clone the following repository: [Github](https://github.com/ethanfetaya/NRI)
- Follow the steps mentioned in the repo to generate the dataset.
- All the normal files replicate the results using Gumbel method.
- `*_cat` replicate using the Categorical method.
- `*_scat` replicate using the Categorical++ method.

Run the following command to simulate the experiment:
```bash
python train.py --epochs 100 --encoder cnn --decoder rnn 
```
Use the train file as per your estimator