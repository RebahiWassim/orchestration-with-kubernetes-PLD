# orchestration-with-kubernetes-PLD

1-Construire l’image

C:\Users\ShiftTech\Desktop\S8\PLD\bone-cancer> docker build -t wassimrebahi/bone-model:v1 .

2-verification(optionnel) :

docker images

3-Teste:

docker run -p 8000:8000 wassimrebahi/bone-model:v1

4-push dans le dockerhub

docker push wassimrebahi/bone-model:v1

5-install minikube & ajouter le path dans le variable d'environnement : https://minikube.sigs.k8s.io/docs/start/

6- installe kubectl & ajouter le path dans le variable d'environnement

(en mode administrateur m7abatch tmchi en mode normal)

7- start minikube : minikube start

8- configuration avec kubectl : kubectl config use-context minikube

9-verification des neouds: kubectl get nodes

10- creer fichier deploiement.yml pour kubernetes:

C:\Windows\system32>nano liver-deployment.yaml

le contenu comme ça : 
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liver-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: liver-model
  template:
    metadata:
      labels:
        app: liver-model
    spec:
      containers:
        - name: liver-model
          image: yourdockerhub/liver-model:v1
          ports:
            - containerPort: 8000

11- ajouter comme pod sur kubernetes :
C:\Windows\system32>kubectl apply -f liver-deployment.yaml

12-verification : kubectl get pods ( status should be running it takes a time few minutes)

13- la creation du fichier service : C:\Windows\system32>nano liver-service.yaml
contenu:
apiVersion: v1
kind: Service
metadata:
  name: liver-service
spec:
  type: NodePort
  selector:
    app: liver-model
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30007

14- creation du service sur kubernetes :
C:\Windows\system32>kubectl apply -f liver-service.yaml

15- verification : kubectl get services

16- lancer le service :
C:\Windows\system32>kubectl get services
NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
kubernetes      ClusterIP   10.96.0.1        <none>        443/TCP        77m
liver-service   NodePort    10.103.239.187   <none>        80:30007/TCP   38m

C:\Windows\system32>minikube service liver-service
┌───────────┬───────────────┬─────────────┬───────────────────────────┐
│ NAMESPACE │     NAME      │ TARGET PORT │            URL            │
├───────────┼───────────────┼─────────────┼───────────────────────────┤
│ default   │ liver-service │ 80          │ http://192.168.49.2:30007 │
└───────────┴───────────────┴─────────────┴───────────────────────────┘
* Tunnel de démarrage pour le service liver-service.
┌───────────┬───────────────┬─────────────┬────────────────────────┐
│ NAMESPACE │     NAME      │ TARGET PORT │          URL           │
├───────────┼───────────────┼─────────────┼────────────────────────┤
│ default   │ liver-service │             │ http://127.0.0.1:54569 │ put this in navigator with /docs
└───────────┴───────────────┴─────────────┴────────────────────────┘
* Ouverture du service default/liver-service dans le navigateur par défaut...
! Comme vous utilisez un pilote Docker sur windows, le terminal doit être ouvert pour l'exécuter.
* Tunnel d'arrêt pour le service liver-service.


16- duplique le service (optionnel pour la possibilité de la charge sur le service) :
C:\Windows\system32>kubectl scale deployment liver-model --replicas=5

tout ça pour un seul model!!!!


dans le cas vous avez fait une modification au niveau de fichier fast api vous avez besoin de execute les cmd suivant :
docker build -t wassimrebahi/liver-model:v3 .
docker push wassimrebahi/liver-model:v3

change l'image dans le fichier liver-deployement.yml:
image: wassimrebahi/liver-model:v3


kubectl apply -f liver-deployment.yaml
kubectl rollout status deployment/liver-model
