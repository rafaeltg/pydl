# PYDL CLI

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
          "path": "/path/to/your/train/x/csv/with/header",
          "params": {
            "has_header": true
          },
          "train_y": {
            "path": "/path/to/your/train/y/csv/with/header",
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
    python3.5 run.py fit -c mlp_fit.json -o result/folder/
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
          "path": "/path/to/your/test/x/csv/with/header",
          "params": {
            "has_header": true
          }
        }
      }
    }
    ```
* Command: 
    ```
    python3.5 run.py predict -c mlp_predict.json -o result/folder/
    ```
* Output:
    * .npy file with the predictions (mlp_test_preds.npy)
    
### predict_proba
### score
### transform
### reconstruct
### cv
### optimize