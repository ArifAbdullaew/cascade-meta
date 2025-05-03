# Running Cascade Fuzzer in a Local Cluster with Docker & Ray

This guide provides a complete step-by-step setup for running the [Cascade CPU fuzzer](https://github.com/cascade-artifacts-designs/cascade-meta) across multiple Docker containers within a local network. It includes pulling a pre-built image, creating a shared NFS directory, setting up a custom Docker network, and deploying a Ray-based cluster for distributed fuzzing.


## 1. Pull the Cascade Docker Image

Cascade provides a pre-configured Docker image containing a ready-to-use fuzzing environment. Pull it on each host using:

```bash
docker pull docker.io/ethcomsec/cascade-artifacts
```

## 2. Set Up a Shared NFS Directory

### On the NFS Server (Master Host):

```bash
sudo apt update && sudo apt install nfs-kernel-server -y
sudo mkdir -p /shared/data/cascade-data
sudo chmod -R 777 /shared/data/cascade-data
echo "/shared/data/cascade-data *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

### On the NFS Clients (Other Hosts):

```bash
sudo apt update && sudo apt install nfs-common -y
sudo mount <master-ip>:/shared/data/cascade-data /mnt/cascade-data
```

Replace `<master-ip>` with the actual IP address of the NFS server.

## 3. Create a Docker IPvLAN Network

This allows each container to receive its own IP address in the physical network:

```bash
docker network create -d ipvlan   --subnet=192.168.1.0/24   --gateway=192.168.1.1   -o parent=eth0   my_ipvlan
```

## 4. Run Containers with Fixed IP and NFS Mount

### On the First Host:

```bash
docker run -dit --name cascade-container   --network my_ipvlan --ip 192.168.1.100   -v /shared/data/cascade-data:/cascade-data   ethcomsec/cascade-artifacts
```

### On the Second Host:

```bash
docker run -dit --name cascade-container   --network my_ipvlan --ip 192.168.1.101   -v /mnt/cascade-data:/cascade-data   ethcomsec/cascade-artifacts
```

## 5. Prepare the Environment Inside the Containers

Once inside the container, run the following commands:

## 6. Run the Setup Script:

```bash
chmod +x setup.sh
./setup.sh --role master
```

### On the Worker Node:

Same steps as above, but the last line should be:

```bash
./setup.sh --role worker --master-ip 192.168.1.100
```

## Verify the Cluster

You can verify that the cluster is working by comparing the number of CPUs inside the container and within the Ray dashboard.

Access the Ray dashboard at `http://192.168.1.100:8265` to view cluster status and node information.
