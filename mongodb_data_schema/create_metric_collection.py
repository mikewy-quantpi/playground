from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = client['playground']  # Replace 'my_database' with your database name
collection = db['metric']

# Define the document to insert
document = {
    'name': 'metric_name',
    'value': 10,
    'timestamp': '2024-04-24T12:00:00Z'
}

# Insert the document into the collection
insert_result = collection.insert_one(document)

# Print the inserted document's ID
print("Inserted document ID:", insert_result.inserted_id)

# Close the MongoDB connection
client.close()
