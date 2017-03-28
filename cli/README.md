# pydl CLI

## Available Commands

### fit
* Input (mlp_fit.json):
    ```
    {
      "model": {
        "class_name": "MLP",
        "config": {
          "name": "mlp_test",
          "layers": [50, 50, 25],
          "dropout": 0.2
        }
      },
    
      "data_set": {
        "train_x": {
          "path": "/path/to/your/train_x_with_header.csv",
          "params": {
            "has_header": true
          },
          "train_y": {
            "path": "/path/to/your/train_y_with_header.csv",
            "params": {
              "has_header": true
            }
          }
        }
      }
    }
    ```
* Command: 
    ```
    pydl fit -c mlp_fit.json -o result/folder/
    ```
* Output:
    * .h5 file with model weights (mlp_test.h5)
    * .json file with model configuration (mlp_test.json)
    
### predict
* Input (mlp_predict.json):
    ```
    {
      "model": "mlp_test.json",
      "data_set": {
        "data_x": {
          "path": "/path/to/your/test_x_with_header.csv",
          "params": {
            "has_header": true
          }
        }
      }
    }
    ```
* Command: 
    ```
    pydl predict -c mlp_predict.json -o result/folder/
    ```
* Output:
    * .npy file with the predictions (mlp_test_preds.npy)
    
### predict_proba
* Input (mlp_predict_proba.json):
    ```
    {
      "model": "mlp_test.json",
      "data_set": {
        "data_x": {
          "path": "/path/to/your/test_x_with_header.csv",
          "params": {
            "has_header": true
          }
        }
      }
    }
    ```
* Command: 
    ```
    pydl predict_proba -c mlp_predict_proba.json -o result/folder/
    ```
* Output:
    * .npy file with the predicted probabilities (mlp_test_pred_probas.npy)
    
### score
* Input (mlp_score.json):
    ```
    {
      "model": "mlp_test.json",
      "data_set": {
          "data_x": {
            "path": "/path/to/your/test_x.npy"
          },
          "data_y": {
            "path": "/path/to/your/test_y.npy"
          }
        }
    }
    ```
* Command: 
    ```
    pydl score -c mlp_score.json -o result/folder/
    ```
* Output:
    * .json file with the score function (loss function) value (mlp_test_score.npy)
    
### transform
* Input (ae_transform.son):
    ```
    {
      "model": "ae_test.json",
      "data_set": {
          "data_x": {
            "path": "/path/to/your/test_x.npy"
          }
        }
    }
    ```
* Command: 
    ```
    pydl transform -c ae_transform.json -o result/folder/
    ```
* Output:
    * .npy file with the encoded values of data_x (test_x_encoded.npy)
    
### reconstruct
* Input (ae_reconstruct.son):
    ```
    {
      "model": "ae_test.json",
      "data_set": {
          "data_x": {
            "path": "/path/to/your/test_x_encoded.npy"
          }
        }
    }
    ```
* Command: 
    ```
    pydl reconstruct -c ae_reconstruct.json -o result/folder/
    ```
* Output:
    * .npy file with the reconstructed values of data_x (test_x_encoded_rec.npy)

### cv
* Input (mlp_cv.json):
    ```
    {
      "model": {
        "class_name": "MLP",
        "config": {
            "name": "mlp_test",
            "layers": [32, 16]
        }
      },
      
      "cv": {
        "method": "kfold",
        "params": {
            "n_splits": 5,
        },
        "scoring": ["rmse", "mape"]
      },
        
      "data_set": {
        "data_x": {
            "path": "/path/to/your/train_x.npy"
        },
        "data_y": {
            "path": "/path/to/your/train_y.npy"
        }
      }
    }
    ```
* Command: 
    ```
    pydl cv -c mlp_cv.json -o result/folder/
    ```
* Output:
    * .json file with the values (for each cv fold), mean and standard deviation of each scoring (mlp_test_cv.json)
    
### optimize
### evaluate