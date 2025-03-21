### Exercises from Introduction to Deep Learning course from TUM
- https://dvl.in.tum.de/teaching/i2dl-ss22/ 

### Relevant Materials 
- Stanford Course in Parallel(cs231n) : http://cs231n.stanford.edu/
- Latex Shared Communal Notes : https://sharelatex.tum.de/2973642517rvqschxwndzf


### Multidimensional Derivatives ( Regarding Exercise 04 ) 
A core concept of training a neural network lies in our ability to derive multidimensional tensors efficiently and in parallel, using the great power of libraries, such as NumPy and later on PyTorch. 
But, the moment that we move from doing simple math operations on scalars to matrices, things could get quite complex.
In Exercise_04 you're going to implement your very first simple neural network of this course. However, as simple as it is, understanding how to derive the relevant layers, such as the Linear (fully-connected), sigmoid, and loss functions is super crucial and not always as trivial as one might think.
Therefore, we've assembled for you some reading material that will make your life MUCH easier. We HIGHLY recommend you to give a deep read of the next articles, and even watch the prerecorded video that we've made last semester, regarding the issue.
- Stanford article: [http://cs231n.stanford.edu/handouts/linear-backprop.pdf](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
- TUM article: [TUM___I2DL___Matrix_derivatives.pdf ( ](https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fl2esy9j0tfk2x1%2F540a8caf6acb51a65119f69201d4a4e4a5b0d13f799848c8cf38ed09b4d6624d%2FTUM___I2DL___Matrix_derivatives_%281%29.pdf)
  - https://www.youtube.com/watch?v=e76kJ-Ubl1o we go over the TUM article

### Autoencoder ( Regarding Exercise 08 )

After establishing some roots in the soil of the deep learning world, it’s about time to get down to business! (and defeat the Huns...)

In exercise 08 we’re going to implement one of the most interesting network architectures, in my opinion: the autoencoder! This model is such a powerful tool, that later on was perfected, and it is being used intensively in the industry and research, for the tasks of segmentation, 3D reconstruction, generative models, style transfers, etc. Basically, it has endless applications. To the infinity - and beyond!

Therefore, I would like to point out some important anecdotes:

![Image Placeholder](image.png)

1. The basic architecture is still based on fully connected layers (Of course with activations layers).
2. It consists of two main parts:
   - **Encoder**: Reducing the dimensionality towards the “latent” space forces the network to focus on collecting the most meaningful features from the input data, so the loss of information would be as small as possible. This is a common core idea, such as in a PCA. Therefore, we could say that the Encoder is used as a **features extractor**, much like the upcoming convolutional layers (But yet, not as strong, and still much more expansive).
   - **Decoder**: Receives the dimensionality-reduced latent space, and aims to **reconstruct** the input of the Encoder. The output has the same shape as the input. A crucial note, for the more advanced usage of tasks, such as semantic segmentation.

3. For the loss function, we could use the *L1* loss or the *MSE* (*L2*), between each pixel in the input and its corresponding pixel in the output.

**Latent space:**
- Resides between the encoder and the decoder. It is what we also call a **bottleneck**, because everything has to squeeze and fit through it. Like the Persian army, charging at only 300 Spartans at the battle of Thermopylae.
  - If the size of it is too small, not much information could eventually pass through to the decoder, and the reconstruction would be very hard. The result would be very blurry.
  - If too big, the network could basically learn to copy the image, without learning any meaningful features.

But why do we even want to use that? Autoencoders, as used in exercise 08, is an excellent solution to a state where our dataset is very big, but only just a small part of it is actually labeled, like a medical CT dataset, for example. Or, as in the case of our exercise - when we have a very small dataset, to begin with. So, we will have 2 steps:

1. **Autoencoder → reconstruct the input.** Let the Encoder learn the relevant features about the **unlabeled** data. This part can be referred to as **unsupervised learning**.
2. After the training has converged, remove the decoder and discard it. Then, plug in instead, just after the latent space, a very simple fully-connected classifier, and train on the **labeled data**, given the fact that the remaining Encoder is already trained as a good features extractor. This part can be referred to as **supervised learning**.

And voila! You’ll see that after struggling to achieve some meaningful classification scores for CIFAR-10 in the past two exercises, you now suddenly will get accuracy percentages as high as in the 70s and even 80s, although the MNIST is a much simpler dataset than Cifar-10.



--- 

### Deep learning: Pipeline

Now that we've touched all the important steps of deep learning pipeline, I would like to do a recap.  
This post is basically meant for later, when you're smarter and older - as it gives some overview of the pipeline, from actual experience, that has been learned in blood, toil, tears and sweat.

**0. Do we even need a deep-learning application?**
If our problem is Linear, just use linear regression, which has a closed form solution.

**1. Collect data**
This is a very not trivial step, and although that within this course we work with some fun datasets, such as Cifar-10, MNIST, etc - real world data is really scarce, and usually there just isn't one available to our needs. It means we will have to collect the data in massive quantities, filter it, annotate it - or maybe apply some techniques to extract relevant information from it in an unsupervised fashion (exercise 08).

> NOTE: Also, this step is bound to human errors. Therefore, when you first put your hands on a new dataset, you MUST play with it and verify it is actually appropriate for the task at hand and that the ground-truth data, from which the model is going to learn features of the real world, is actually accurate. For example, take this image from the Raidar dataset (https://raidar-dataset.com/), which offers ground truth for 2D images, for the task of semantic segmentation (exercise_10), where we predict a label for each pixel, and not only for the entire image.

These samples are bad, and will make false predictions on the test set.

<table>
    <tr>
        <td><img src="https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fl2esy9j0tfk2x1%2F0695a137837e7f843e82125862cae81225d5b5135d70ed9954bfcc78cfcfcc37%2Fimage.png" width="100%"></td>
        <td><img src="https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fl2esy9j0tfk2x1%2F7f51aeacd3f68c34c5fb9b3ec0fde20dae5ce67b1bbdbbb84fbaa846a81d1a68%2Fimage.png" width="100%"></td>
    </tr>
</table>


**2. Datasets and Dataloaders (exercise_03):**
While we tend to forget about this stage, it is a corner stone for a well behaving training session. Once you've verified that your dataset is good to work with, it is time to load it. Frameworks such as PyTorch have builtin functionalities, that will make your life really easy.

However, it is important to remember the structures:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, some_arguments) -> None:
        super().__init__()
        # Here you keep:
        # - Files list
        # - Define Transormations, such as FLIP, ToTensor, Noramlization, etc.
        # What kind of split is it? Train \ Val \ Test.
       
    def __getitem__(self, index):
        # Get a SINGLE instance form the dataset, 
        # which is initiated in __init__().

        # Usually we load here the image from disk, or access it if it is already in the RAM.
        # Apply relevant transformations and data augmentations.

        # I recommend always retuning a dictionary, 
        # as it is really comfortable to work with.
        return {"image": x, "gt": y, "name": file_name}
    
    def __len__(self):
        return len(self.file_list)
```

PyTorch would take such class, and create a give you a dataloader that you could use.

```python
from torch.utils.data import DataLoader

def make_dataloaders(split="train", train_part=-1, num_samples=-1, batchsize=-1):
    
    """
    Create train and val dataloaders out of the split (Given by nuscenes)
    """
    
    batchsize = params.batch_size if batchsize < 1 else batchsize
    params.num_samples = num_samples if num_samples > 0 else params.num_samples
    files_list = load_prepared_file_list(split)
    files_list = files_list[0: params.num_samples] if split == "train" else files_list[params.num_samples: params.num_samples + params.num_samples_offset]
    dataset = MyDataset(list_with_all_files=files_list)
    fractions = [int(round(train_part * params.num_samples)), int(round((1 - train_part) * params.num_samples))]  if 0 < train_part < 1 else params.train_val_split
    train_ds, val_ds = torch.utils.data.random_split(dataset, fractions) 
    train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=params.num_workers_train)
    val_dl = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=params.num_workers_eval)
    return train_dl, val_dl

# Overfitting simply:
train_dl, val_dl = make_dataloaders(num_samples=100, train_part=0.8)  
```

Pay attention to the `num_workers` argument, that the PyTorch class `Dataloader` takes, as if your hardware can allow it, a higher number would mean that more threads work in parallel to load your data, hence the training epoch time could drop from 7 hours to 3 hours per epoch, which is substantial.

Now, all is left is to use it within your training program:

```python
loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)
for i, batch in loop:
    ...
```

**3. Cross-validation and hyperparameters turning.**
Note that this a bit of the "chicken and the egg" problem - what comes first?  
As cross validation is mostly there to ensure that our splits of train / val are balanced (the fair distribution of hard and easy, day and night, winter and summer, etc), I would say that you shuffle the data, fix train and val splits, perform hyperparameters tuning, and only then perform the cross-validation step, to fine tune.

**4. Model and Loss function.**
The bread and butter of the deep learning process. Usually when we start a deep learning project in Uni, we rely on an already existing model, that was carefully thought of. However, as it was created by humans, it could still contain errors. Never take a code base as a guarantee for perfection. You should read it thoroughly, understand the choices in architecture, and only then decide if to go with it, or tweak some stuff. It might be a wearing process, but it will save you SO MUCH time later on, and after all - you need to really understand what's going on, if you wanna add something or even debug your code.

You should have a STRONG understanding of the layers we introduce in this course, such:

- Convolutions  
- Batch Normalization (and Group-Normalization, as it comes really handy in real world applications, where the batch size is really small).  
- Dropout  
- Activation functions and where to find them.  

Loss functions: MSE and L1 vs HuberLoss, Crossentrpy vs FocalLoss, etc. Note that regression loss functions (MSE or L1) DO NOT couple with a corresponding activation function, such as softmax or sigmoid.  

Also, the usage of Skip-connections (exercise 10) is neglectable in memory and computational load and therefore is SUPER recommended. Note that for convolutions is it better to use concatenation in the channels' axis than element-wise summation.

**5. Training process tips**

- The core pipline: forward pass → loss calculation → backward pass → optimizer step  
- We usually validate the network after each epoch.  
- Use a configuration file, e.g. `params.py`, where you set all your hyperparameters, global variables, and magic numbers.  
- ALWAYS use `graidnet_clip`. This is such a simple thing, that prevents your weights from exploding.  
- When using PyTorch, don't forget to use `optimize.zero_grad()`, to clear your gradient at the beginning of each iteration.  
- If the batch_size is small, you could perform a few iterations before updating the weights.

```python
if (i + 1) % params.optim_update_num_iter == 0 or (i + 1) == len(self.train_dataloader):
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
```

- Use a scheduler, where the most basic one is the learning_rate decay process.  
- Always use Early Stopping  
- Make sure not to accumelate variables into the GPU ram. For example, if saving losses to some list, always convert the loss tensor to CPU variable, by:

```python
loss = nn.CrossEntropy()(pred, gt)
cpu_loss = loss.item()
```
![Training Loss Graph](https://cdn-uploads.piazza.com/paste/l2esy9j0tfk2x1/c34e7995727cfe99bc9eaad7e9e07334cfd91da43c7ddd1b1a70df0b9a031f3f/image.png)

- Use tqdm to print the progress of your training, and update it in the loop.

```python
from tqdm import tqdm

loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)
for i, batch in loop:
  ...
  loop.set_description(f'Epoch [{epoch}/{params.num_epochs}]')
  loop.set_postfix(curr_loss = loss_cpu.item(),
                   learning_rate = "{:.7f}".format(cur_lr),
                   running_mean =  "{:.6f}".format(all_losses_mean),
                   last_50_mean =  "{:.6f}".format(recent_losses_mean),
```

- Tensorboard (exercise 07) - what an amazing tool. You also have Weights and Biases, but it might be much more complex for noobies like you ;)

Just make sure you have the latest version, and then you could easily save your progress, and cheer for it as it is going down!

In your terminal run:  
```bash
tensorboard --logdir <where_to_look_for_tesnsorboard_files>
```



```python
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    def __init__(self, output=True) -> None:
        self.tb_logger = SummaryWriter(self.new_log_path, flush_secs=10)

    def train_one_epoch(self, epoch, training_steps, save=True):
        ....
        if save:
            self.tb_logger.add_scalar("Loss/train", loss, training_steps)
```

**DON'T FORGET TO SAVE CHECKPOINTS:**

Well, you're going to have to be patient and let your baby train.  
But what if the process dies in the middle? What if you want to stop and try something else, but it's a shame to go through all the initial training all over again?

Don't worry! PyTorch has got your back. After each training epoch, we validate our performance on the validation set. Then, if a new best lowest validation loss has been achieved (I usually do it by taking the mean of all the validation losses), save your model. Create an output, and save your outputs to it:

```python
if eval_loss < best_eval_loss:
    best_eval_loss = eval_loss
    if save:
        state = {'state_dict': self.model.to('cpu').state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, os.path.join(self.new_run_path, "mlt" + '_epoch_' + str(epoch+1) +"_best_eval_loss_" + "{:.5f}".format(best_eval_loss.item()) +'.pth'))
        self.model.to(device=self.device)
        print('Model saved.')
```

![checkpoint](https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fl2esy9j0tfk2x1%2Fda8df18ccb1d098e83f190ce0a4792f67e9421a2f27b9b9ac35704ee933cab71%2Fimage.png)


Then, at the beginning of a new session, or for evaluation, you could simply load your model (It HAS to be the same model, with no changes)

```python
self.model = MyPyTorchModel().to(self.device)

if params.checkpoint is not None and os.path.exists(params.checkpoint):
    state = torch.load(params.checkpoint)
    self.model.load_state_dict(state["state_dict"])
    self.model.to(self.device)
    params.learning_rate = state.get("lr", params.learning_rate)
```

Also, with early stopping, you are secure - and know that your best model has been saved before the model stopped.  
Go, and have fun making the world a better place! :)


