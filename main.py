import train_model
import optuna
from functools import partial

def objective(trial, base_config):
    # 定义超参数搜索空间
    config = {
        'beta': trial.suggest_float('beta', 0.5, 3.0),
        'num_prune': trial.suggest_int('num_prune', 1, 7),
        'prune_interval': trial.suggest_int('prune_interval', 2, 10),
        'num_epochs': base_config['num_epochs'],
        'is_prune': base_config['is_prune'],
        'prune_layer1': base_config['prune_layer1'],
        'prune_layer2': base_config['prune_layer2'],
    }
    
    # 修改train_model函数返回最佳性能指标
    final_metric = train_model(config)
    return final_metric

if __name__ == '__main__':
    base_config = {
        'prune_layer1': True,
        'prune_layer2': True,
        'num_epochs': 50,
        'is_prune': True
    }
    
    # 创建study对象
    study = optuna.create_study(
        direction="maximize",
        study_name="cnn_optimization",
        storage="sqlite:///study.db",  # 保存优化结果到数据库
        load_if_exists=True
    )
    
    # 运行优化
    study.optimize(
        partial(objective, base_config=base_config),
        n_trials=20,  # 优化试验次数
        timeout=None  # 可选：设置总时间限制（秒）
    )
    
    # 打印优化结果
    print("最佳参数:", study.best_params)
    print("最佳性能:", study.best_value)
    
    # 可视化优化过程
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_parallel_coordinate(study)