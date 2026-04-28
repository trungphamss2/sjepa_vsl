import numpy as np

def geometric_transformation(skeleton, angle_range=30, p_flip=0.5):
    """
    Biến đổi hình học chuẩn S-JEPA:
    1. Xoay Rodrigues quanh trục xương sống.
    2. Lật gương trái-phải.
    3. Dịch chuyển tịnh tiến nhẹ.
    """
    T = skeleton.shape[0]
    
    # Rodrigues Rotation
    pelvis = skeleton[:, 0, :]
    shoulder_mid = skeleton[:, 20, :]
    n_vector = shoulder_mid - pelvis 
    n = np.mean(n_vector, axis=0)
    n = n / (np.linalg.norm(n) + 1e-8)
    
    alpha = np.radians(np.random.uniform(-angle_range, angle_range))
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    nx, ny, nz = n
    K = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
    I = np.eye(3)
    R = I + sin_a * K + (1 - cos_a) * np.dot(K, K)
    skeleton_rot = np.dot(skeleton, R.T)
    
    # Spatial Flip
    if np.random.random() < p_flip:
        skeleton_rot[:, :, 0] = -skeleton_rot[:, :, 0]
        pairs = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (21, 23), (22, 24)]
        for left, right in pairs:
            temp = skeleton_rot[:, left, :].copy()
            skeleton_rot[:, left, :] = skeleton_rot[:, right, :]
            skeleton_rot[:, right, :] = temp
            
    # Jitter Translation
    jitter = np.random.uniform(-0.02, 0.02, size=(1, 1, 3))
    skeleton_final = skeleton_rot + jitter
    return skeleton_final.reshape(T, -1)
