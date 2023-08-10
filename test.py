import torch

def custom_interpolate(input_tensor, target_size, mode='bilinear'):
    # Calculate scaling factors
    scale_factor_rows = input_tensor.shape[2] / target_size[0]
    scale_factor_cols = input_tensor.shape[3] / target_size[1]
    
    # Initialize output tensor
    output_tensor = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], target_size[0], target_size[1])
    
    for row in range(target_size[0]):
        for col in range(target_size[1]):
            # Calculate the corresponding indices in the input tensor
            row_idx = row * scale_factor_rows
            col_idx = col * scale_factor_cols
            
            # Get the surrounding pixel indices
            top_left_row = int(row_idx)
            top_left_col = int(col_idx)
            
            # Calculate the interpolation weights
            weight_row = row_idx - top_left_row
            weight_col = col_idx - top_left_col
            
            # Perform interpolation using bilinear formula
            top_left_value = input_tensor[:, :, top_left_row, top_left_col]
            top_right_value = input_tensor[:, :, top_left_row, min(top_left_col + 1, input_tensor.shape[3] - 1)]
            bottom_left_value = input_tensor[:, :, min(top_left_row + 1, input_tensor.shape[2] - 1), top_left_col]
            bottom_right_value = input_tensor[:, :, min(top_left_row + 1, input_tensor.shape[2] - 1), min(top_left_col + 1, input_tensor.shape[3] - 1)]
            
            interpolated_value = (1 - weight_row) * (1 - weight_col) * top_left_value + \
                                 (1 - weight_row) * weight_col * top_right_value + \
                                 weight_row * (1 - weight_col) * bottom_left_value + \
                                 weight_row * weight_col * bottom_right_value
            
            output_tensor[:, :, row, col] = interpolated_value
    
    return output_tensor

# Test the custom function
def test_custom_interpolate():
    input_tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
    target_size = (2, 2)
    
    custom_result = custom_interpolate(input_tensor, target_size)
    torch_result = torch.nn.functional.interpolate(input_tensor, target_size, mode='bilinear', align_corners=False)
    
    assert torch.allclose(custom_result, torch_result, rtol=1e-3)

test_custom_interpolate()
