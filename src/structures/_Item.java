package structures;

import java.util.ArrayList;

/**
 * @author Lu Lin
 * The structure is used to store item information
 */
public class _Item {
    protected String m_itemID;

    protected ArrayList<_Review> m_reviews;

    public double[] m_eta; // variational inference for p(\gamma|\eta)

    public _Item(String itemID){
        m_itemID = itemID;
        m_reviews = null;
    }
}
