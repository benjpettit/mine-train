from flask import request
from flask_api import FlaskAPI
import sys
import os
from app.learner import Learner
app = FlaskAPI(__name__)

@app.route("/nextItems", methods=['GET'])
def getNextItems():
    return learner.getCandidates()

@app.route("/label", methods=['POST'])
def processLabel():
    """
    Handle ids labelled by the user:
    """
    docId = request.data["id"]
    label = int(request.data["liked"])
    responseMessage = learner.handleResponse(docId, label)
    return {"message": responseMessage}

if __name__ == "__main__":
    dataDir = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except:
        port = 5000
    learner = Learner(dataDir)
    app.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)



