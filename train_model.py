import torch
import sys
import os
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data_utils import raw_taxi_df, clean_taxi_df, split_taxi_data
from model_utils import MLP, NYCTaxiExampleDataset

def main(num_epochs:int = 5, lr = 1e-4, train_batch_size = 10):
    """
    Simple training loop
    Args:
        num_epochs(int): number of epochs to train the mlp; 
        lr: learning rate of Adam; 
        train_batch_size: batch size used in the Dataloader.
    """

    # Set fixed random number seed
    torch.manual_seed(42)
  
    """
    data loading and cleaning:
    clean and process data by:
        (1) drop all the NA values
        (2) remove trips with distance greater than 100 (possible outliers)
        (3) pick out the pickup locations and drop-off locations as input features to predict the fare amount
    """
    raw_df = raw_taxi_df(filename="data/yellow_tripdata_2024-01.parquet")
    clean_df = clean_taxi_df(raw_df=raw_df)
    location_ids = ['PULocationID', 'DOLocationID']
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                   x_columns=location_ids, 
                                                   y_column="fare_amount", 
                                                   train_size=500000)

    # Pytorch
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
  
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
  
    # Run the training loop
    for epoch in range(0, num_epochs): # can modify the number of epochs
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10000 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            if i == len(trainloader) - 1:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

    # save model to the 'models' folder
    torch.save({'model_state_dict': mlp.state_dict(),}, os.path.join(os.path.dirname(__file__), 'models'))

    return X_train, X_test, y_train, y_test, data, mlp

if __name__ == "__main__":
    main()

