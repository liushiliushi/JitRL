from jericho import *
import sys

class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, step_limit=None, get_valid=True, cache = None):
        self.rom_path = rom_path
        self.env = FrotzEnv(rom_path)
        self.bindings = self.env.bindings
        self.seed = seed
        self.steps = 0
        self.end_scores = []
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.cache = cache
        
        self.use_parallel = not self._is_debugging()
        

    def get_objects(self):
        desc2objs = self.env._identify_interactive_objects(use_object_tree=False)
        obj_set = set()
        for objs in desc2objs.values():
            for obj, pos, _ in objs:
                if pos == 'ADJ': continue
                obj_set.add(obj)
        return list(obj_set)

    def _is_debugging(self):
        """Detect if running in debug mode."""
        debugger_indicators = [
            'pydevd',  # PyCharm/VSCode debugger
            'debugpy',  # VSCode Python debugger
            'pdb',     # Python debugger
            'ipdb',    # IPython debugger
        ]
        
        for indicator in debugger_indicators:
            if any(indicator in module_name for module_name in sys.modules):
                return True
        return False

    def _get_state_hash(self, ob):
        return self.env.get_world_state_hash()
    

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:
            save = self.env.get_state()
            hash_save = self._get_state_hash(ob)
            if self.cache is not None and hash_save in self.cache:
                info['look'], info['inv'], info['valid'] = self.cache[hash_save]
            else:
                look, _, _, _ = self.env.step('look')
                info['look'] = look.lower()
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv.lower()
                self.env.set_state(save)
                # Get the valid actions for this state
                if self.get_valid:
                    valid = self.env.get_valid_actions(use_parallel=self.use_parallel)
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = valid
                if self.cache is not None:
                    self.cache[hash_save] = info['look'], info['inv'], info['valid']
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: 
            self.end_scores.append(info['score'])
        return ob.lower(), reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        valid = self.env.get_valid_actions(use_parallel=self.use_parallel)
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        return initial_ob, info


    def copy(self):
        copy_env = JerichoEnv(self.rom_path, self.seed)
        copy_env.env = self.env.copy()
        copy_env.cache = self.cache
        return copy_env
    

    def close(self):
        self.env.close()