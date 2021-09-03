from azureml.core import ScriptRunConfig, Experiment
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.widgets import RunDetails
from azureml.core.authentication import ServicePrincipalAuthentication

svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxx-xxxxxx-xxxx-xxxx-xxxxxxx",
    service_principal_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxxx")

ws = Workspace(
    subscription_id="xxxxxxxx-xxxxxxx-xxxx-xxx-xxxxxxxxx",
    resource_group="rg-machinelearning",
    workspace_name="machinelearning",
    auth=svc_pr
    )

print("Found workspace {} at location {}".format(ws.name, ws.location))

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# create or load an experiment
experiment = Experiment(ws, 'MyExperiment')

# create or retrieve a compute target
cluster = ws.compute_targets['cluster1']

# create or retrieve an environment
env = Environment.get(ws, name='AzureML-sklearn-0.24.1-ubuntu18.04-py37-cpu-inference')

# configure and submit your training run
src = ScriptRunConfig(source_directory='.',
                        command=['bash setup.sh && python train.py'],
                        compute_target=cluster,
                        environment=env)

run = experiment.submit(config=src)
run