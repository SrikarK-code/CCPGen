def train_mfm(mfm, scvi_latents, prot_latents, n_epochs=1000, batch_size=128, lr=1e-4):
    optimizer_eta = Adam(mfm.interpolant.parameters(), lr=lr)
    optimizer_theta = Adam(mfm.vector_field.parameters(), lr=lr)
    
    mfm.metric.train_metric(prot_latents)
    coupling = compute_ot_coupling(scvi_latents, prot_latents)
    
    for epoch in range(n_epochs):
        # Step 1: Train interpolants
        indices = torch.multinomial(coupling.flatten(), batch_size, replacement=True)
        i, j = indices // coupling.shape[1], indices % coupling.shape[1]
        x0, x1 = scvi_latents[i], prot_latents[j]
        
        loss_geodesic = mfm.compute_geodesic_loss(x0, x1)
        optimizer_eta.zero_grad()
        loss_geodesic.backward()
        optimizer_eta.step()
        
        # Step 2: Train vector field
        loss_mfm = mfm.compute_mfm_loss(x0, x1)
        optimizer_theta.zero_grad()
        loss_mfm.backward()
        optimizer_theta.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Geodesic Loss: {loss_geodesic.item():.4f}, MFM Loss: {loss_mfm.item():.4f}")
