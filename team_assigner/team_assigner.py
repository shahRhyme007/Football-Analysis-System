from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}



    #* get player color 
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the labels for the image using the clustering model
        labels = kmeans.labels_

        # Reshaping the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Getting the player cluster and non player cluster 
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


# --------------------------------------------------------------#

    # * assign team color
    def assign_team_color(self,frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

# --------------------------------------------------------------#
    # * get clustering model
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans


# --------------------------------------------------------------#

    # * get player team 
    def get_player_team(self,frame,player_bbox,player_id):
        
        if player_id in self.player_team_dict:
            # player has already been assigned a team
            return self.player_team_dict[player_id]
        # player has not been assigned a team yet
        player_color = self.get_player_color(frame,player_bbox)

        # get the team id based on the player color
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        # assign the team id to the player dictionary
        if player_id ==91:
            team_id=1
        
        # assign the team id to the player dictionary
        self.player_team_dict[player_id] = team_id

        # return the team id
        return team_id