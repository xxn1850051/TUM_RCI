import csp as AIMAcsp

class TUMCSP(AIMAcsp.CSP):

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())

        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.initial = ()
        self.curr_domains = None
        self.nassigns = 0


    def assign(self, var, val, assignment):
        super(TUMCSP, self).assign(var, val, assignment)
        print( 'Value ' + val + ' assigned to variable ' + var)
        # self.display(assignment)

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print('CSP:', self, 'with assignment:', assignment)

    def display_domain(self):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        for v in self.variables:
            print('variable:', v, 'with domain:', self.choices( v) )

# ______________________________________________________________________________
# Constraint Propagation with AC-3

def AC3(csp, queue=None, removals=None):
    """The AC3 algorithm"""
    if queue is None:
        queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
    csp.support_pruning()
    while queue:
        if make_arc_consistent(csp,queue, removals) == False:
            return False
    return True

def make_arc_consistent(csp,queue=None, removals=None):
    """Check weather the first arc(Xi, Xj) in the queue is arc-consistent or not.
    If not, the Xi is maked arc-consistent qith Xj by pruning the domain of Xi.
    If during pruning the domain becomes empty, the function returns false.
    Otherwise the neighbors of Xi, except Xj, are added t the queue as arcs with Xi.  """
    (Xi, Xj) = queue.pop(0)
    print( 'Checking if ' + Xi + ' is arc-consistent with ' + Xj)
    revised, _ = AIMAcsp.revise(csp, Xi, Xj, removals)
    if revised:
        print( Xi + ' was not consistent with ' + Xj + '. Domain of ' + Xi + ' is now: ' +  "".join(csp.curr_domains[Xi]) )
        if not csp.curr_domains[Xi]:
            return False
        for Xk in csp.neighbors[Xi]:
            if Xk != Xj:
                queue.append((Xk, Xi))
                print('Added to queue (' + Xk + ', ' + Xi + ')' )
    else:
        print( Xi + ' is already arc-consistent with ' + Xj)
    return

# ______________________________________________________________________________
# CSP Backtracking Search

# ______________________________________________________________________________

def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = AIMAcsp.parse_neighbors(neighbors)
    return TUMCSP(list(neighbors.keys()), AIMAcsp.UniversalDict(colors), neighbors,
               AIMAcsp.different_values_constraint)

def australia():
    return MapColoringCSP(list('RGB'),
                        'SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ')
