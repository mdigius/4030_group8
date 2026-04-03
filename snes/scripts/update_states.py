import sys
sys.path.append("/code")
from env import make_env

states = ['TimeTrial.state', 'MarioCircuit_M.state']
game = 'SuperMarioKart-Snes'

for state in states:
    try:
        env = make_env(game=game, state=state, render_mode=None)
        env.reset()
        new_state = env.env.em.get_state()
        env.close()
        
        with open(f"/code/{state}", "wb") as f:
            f.write(new_state)
            
        print(f"Updated {state} accurately.")
    except Exception as e:
        print(f"Error on {state}: {e}")
