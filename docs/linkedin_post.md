My connected-vehicle attack detector caught every crude lie within a second. It caught 0% of the careful ones.

Connected vehicles broadcast their position, speed and acceleration 10 times a second (SAE J2735 Basic Safety Messages). Digital signatures prove who sent a message, not whether it is true, so a compromised vehicle can lie about itself.

I built a roadside detector for that: physics rules plus an isolation forest and an autoencoder, trained only on normal traffic. The normal traffic is real: 677,465 messages built from NGSIM I-80 trajectories with GPS error.

Results on 66 attacking vehicles:
- Fake positions, fake speeds, fake stalled cars, replayed messages: all flagged, median 0.8 s
- A vehicle that drifts its reported position forward smoothly, keeping speed consistent: 0%
- A smooth fake lane change: 17%

The surprise: my first version of the drift attack was "caught" 83% of the time. The detector was reacting to the sudden step in my attack, not the lie. Once the attacker ramped it in over 3 seconds, detection fell to zero.

What I take from it: content checks catch careless attackers. Catching careful ones needs independent evidence, such as the roadside unit's own radar or camera, or what other vehicles report seeing.

Also honest: thresholds set to flag 2% of normal vehicles flagged 4.6% later in the recording, when traffic got heavier.

Repo: https://github.com/faysalchowdhoury/v2x-bsm-fdi-detector

How are V2X misbehaviour detection systems validated in deployments you've seen?

#ConnectedVehicles #V2X #Cybersecurity #AnomalyDetection #TransportationEngineering #MachineLearning #Python
