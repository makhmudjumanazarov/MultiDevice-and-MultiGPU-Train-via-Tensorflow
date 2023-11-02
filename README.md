### Multi-worker configuration

- A configuration
<pre>
 tf_config = {
    'cluster': {'worker': ['192.169.0.146:12345', '192.169.0.128:12345']},
    'task': {'type': 'worker', 'index': 0}
}
tf_config['task']['index'] = 0
os.environ['TF_CONFIG'] = json.dumps(tf_config)
</pre> 
- With terminal command
<pre>
user@User_0$: export TF_CONFIG='{"cluster": {"worker": ["192.169.0.146:12345", "192.169.0.128:12345"]}, "task": {"index": 0, "type": "worker"}}'
</pre> 
</pre> <pre>
user@User_1$: export TF_CONFIG='{"cluster": {"worker": ["192.169.0.146:12345", "192.169.0.128:12345"]}, "task": {"index": 1, "type": "worker"}}'
</pre> 

![Device 0](/device_0.jpg)



